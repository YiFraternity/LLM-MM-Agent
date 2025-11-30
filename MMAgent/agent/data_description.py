from .base_agent import BaseAgent
from prompt.template import DATA_DESCRIPTION_PROMPT
from utils.retry_utils import retry_on_api_error

class DataDescription(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)

    @retry_on_api_error(max_attempts=3,  min_wait=5)
    def summary(self, data_description: str):
        prompt = DATA_DESCRIPTION_PROMPT.format(data_description=data_description)
        return self.llm.generate(prompt)

