from collections import deque
import sys
import logging
from typing import List
from prompt.template import (
    TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT,
    TASK_DEPENDENCY_ANALYSIS_PROMPT,
    DAG_CONSTRUCTION_PROMPT,
)
from utils.retry_utils import (
    LogicError,
    retry_on_api_error,
    reflective_retry_on_logic_error,
    ensure_parsed_json_output,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Coordinator:
    def __init__(self, llm):
        self.llm = llm
        self.memory = {}
        self.code_memory = {}
        self.task_dependency_analysis = None
        self.DAG = None
        self.order = None
        self.task_descriptions = None

    def to_dict(self):
        return {
            "memory": self.memory or {},
            "code_memory": self.code_memory or {},
            "task_dependency_analysis": self.task_dependency_analysis or [],
            "DAG": self.DAG or {},
            "order": self.order or []
        }

    @classmethod
    def from_dict(cls, llm, data: dict):
        instance = cls(llm)
        instance.memory = data.get("memory", {})
        instance.code_memory = data.get("code_memory", {})
        instance.task_dependency_analysis = data.get("task_dependency_analysis", [])
        instance.DAG = data.get("DAG", {})
        instance.order = data.get("order", [])
        return instance

    def compute_dag_order(self, graph):
        """
        Compute the topological sorting (computation order) of a DAG.
        :param graph: DAG represented as an adjacency list, in the format of {node: [other nodes that this node depends on]}.
        :return: A list representing the computation order.
        """
        # Calculate indegree
        in_degree = {node: 0 for node in graph}
        for node in graph:
            in_degree[node] += len(graph[node])

        # Find all nodes with in-degree 0 (which can be used as the starting point for calculation)
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            # Traverse all nodes, find the nodes that depend on the current node, and reduce their in-degree
            for neighbor in graph:
                if node in graph[neighbor]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Check if there is a loop (if the number of sorted nodes is less than the total number of nodes, then there is a loop)
        if len(order) != len(graph):
            raise ValueError("Graph contains a cycle!")

        return order

    @retry_on_api_error(max_attempts=3,  min_wait=5)
    def analyze(self, tasknum: int, modeling_problem: str, task_descriptions: List[str], with_code: bool):
        problem_str = modeling_problem[:5000]
        if with_code:
            prompt = TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT.format(
                tasknum=tasknum,
                modeling_problem=problem_str,
                task_descriptions=task_descriptions
            ).strip()
        else:
            prompt = TASK_DEPENDENCY_ANALYSIS_PROMPT.format(
                tasknum=tasknum,
                modeling_problem=problem_str,
                task_descriptions=task_descriptions
            ).strip()
        return self.llm.generate(prompt)

    @retry_on_api_error(max_attempts=3,  min_wait=5)
    @reflective_retry_on_logic_error(max_attempts=5, wait_time=2)
    @ensure_parsed_json_output
    def dag_construction(self, tasknum: int, modeling_problem: str, task_descriptions: str, task_dependency_analysis: str) -> dict:
        problem_str = modeling_problem[:5000]
        prompt = DAG_CONSTRUCTION_PROMPT.format(
            tasknum=tasknum,
            modeling_problem=problem_str,
            task_descriptions=task_descriptions,
            task_dependency_analysis=task_dependency_analysis
        ).strip()
        return self.llm.generate(prompt)

    def analyze_dependencies(self, modeling_problem: str, task_descriptions: list[str], with_code: bool) -> List[int]:
        task_dependency_analysis = self.analyze(
            len(task_descriptions),
            modeling_problem,
            task_descriptions,
            with_code
        )
        self.task_dependency_analysis = task_dependency_analysis.split('\n\n')
        try:
            self.DAG = self.dag_construction(
                len(task_descriptions),
                modeling_problem,
                task_descriptions,
                task_dependency_analysis
            )
        except LogicError as e:
            logger.error(f"❌ Dependency analysis failed after retries: {e}")
            self.DAG = {}
            return []
        order = self.compute_dag_order(self.DAG)
        return order
