import os
import threading
import queue
import time
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# =========================
# 全局单例（仅在同一进程中生效）
# =========================
_GLOBAL_SCORER = None
_SCORER_LOCK = threading.Lock()


def get_embedding_scorer(model_name, batch_size=4, max_wait=0.01):
    """
    全局只创建一个 EmbeddingScorer，避免在同一进程中多次加载模型引发 OOM。
    （不同 Python 进程之间仍各自加载一次，这由 shell 脚本控制 GPU 分配）
    """
    global _GLOBAL_SCORER
    with _SCORER_LOCK:
        if _GLOBAL_SCORER is None:
            _GLOBAL_SCORER = EmbeddingScorer(
                model_name=model_name,
                batch_size=batch_size,
                max_wait=max_wait,
            )
    return _GLOBAL_SCORER


# =========================
# Dynamic Batcher
# =========================
class EmbeddingBatcher:
    def __init__(self, scorer, batch_size=4, max_wait=0.01):
        self.scorer = scorer
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.q = queue.Queue()

        self.thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.thread.start()

    def submit(self, query, methods):
        """
        外部接口：保持同步调用，内部自动聚合 batch 做一次推理。
        """
        future = queue.Queue(maxsize=1)
        self.q.put((query, methods, future))
        return future.get()

    def _batch_worker(self):
        while True:
            batch = []

            # 先阻塞取一个
            item = self.q.get()
            batch.append(item)

            # 再在 max_wait 时间窗口里尽可能多取
            start = time.time()
            while len(batch) < self.batch_size and time.time() - start < self.max_wait:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    time.sleep(0.001)

            # -------- 构造批次文本 --------
            all_texts = []
            for q, m, future in batch:
                texts = [q] + [
                    f"{mm['method']}: {mm.get('description', '')}" for mm in m
                ]
                all_texts.append(texts)

            flat_texts = sum(all_texts, [])

            # -------- Tokenizer --------
            try:
                inputs = self.scorer.tokenizer(
                    flat_texts,
                    max_length=8192,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.scorer.device)
            except Exception as e:
                print("[Tokenizer Error]", e)
                continue

            # -------- 模型推理（带 OOM 保护） --------
            try:
                with torch.no_grad():
                    outputs = self.scorer.model(**inputs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[OOM] Clearing cache and retrying...")
                    if self.scorer.device.type == "cuda":
                        torch.cuda.empty_cache()
                    time.sleep(0.05)
                    with torch.no_grad():
                        outputs = self.scorer.model(**inputs)
                else:
                    raise e

            embeddings = F.normalize(outputs.last_hidden_state[:, 0], dim=1)

            # -------- 拆分回各个请求 --------
            idx = 0
            for (q, m, future), texts in zip(batch, all_texts):
                n = len(texts)
                emb_slice = embeddings[idx : idx + n]
                idx += n

                query_emb = emb_slice[0:1]
                method_emb = emb_slice[1:]

                scores = (query_emb @ method_emb.T * 100).squeeze().tolist()
                if not isinstance(scores, list):
                    scores = [scores]

                result = [
                    {"method_index": i + 1, "score": float(s)}
                    for i, s in enumerate(scores)
                ]
                future.put(result)


# =========================
# 主类：EmbeddingScorer
# =========================
class EmbeddingScorer:
    """
    使用 HF 模型做语义检索。
    GPU 由外部（shell 脚本）通过 CUDA_VISIBLE_DEVICES 决定，此处只负责使用。
    """

    def __init__(self, model_name, batch_size=4, max_wait=0.01):
        print(">>> Loading embedding tokenizer & model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # 这里不再设置 CUDA_VISIBLE_DEVICES，只尊重外部环境
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()

        self.batcher = EmbeddingBatcher(self, batch_size=batch_size, max_wait=max_wait)

    def score_method(self, query: str, methods: List[dict]) -> List[dict]:
        return self.batcher.submit(query, methods)


# =========================
# 简单测试
# =========================
def _run_single_test(es, idx: int):
    query = f"How to solve modeling problem {idx}?"
    methods = [
        {"method": f"Method-A-{idx}", "description": "This is method A"},
        {"method": f"Method-B-{idx}", "description": "This is method B"},
        {"method": f"Method-C-{idx}", "description": "This is method C"},
    ]
    result = es.score_method(query, methods)
    print(f"[Thread {idx}] result: {result}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("=== Creating EmbeddingScorer ===")
    model_name = os.getenv("EMBED_MODEL")

    # 测试时可以直接 new；实际工程里建议用 get_embedding_scorer()
    es = EmbeddingScorer(model_name, batch_size=4, max_wait=0.02)

    print("=== Launching parallel requests ===")
    threads = []
    for i in range(10):
        t = threading.Thread(target=_run_single_test, args=(es, i))
        t.start()
        threads.append(t)
        time.sleep(0.005)

    for t in threads:
        t.join()
    print("=== Done ===")
