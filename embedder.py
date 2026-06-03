"""文本向量编码器。

默认本地 BGE-small-zh-v1.5（离线、免费、512 维，arm64 走 MPS）。
预留 ApiEmbedder 走 OpenAI 兼容网关的 embeddings 接口作兜底（--backend api）。
所有向量做 L2 归一化，下游用点积即余弦。
"""
import numpy as np

import config

# BGE-zh 检索任务推荐的 query 指令前缀；passage（文档）侧不加。
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _pick_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class LocalBGEEmbedder:
    backend = "local"

    def __init__(self, model_name: str = config.EMB_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.device = device or _pick_device()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:  # 旧版 sentence-transformers
            self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                              batch_size=64, show_progress_bar=len(texts) > 256)
        return v.astype("float32")

    def embed_documents(self, texts) -> np.ndarray:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([_QUERY_PREFIX + text])[0]


class ApiEmbedder:
    """走 OpenAI 兼容网关的 embeddings 接口（需网关支持，opt-in）。"""
    backend = "api"

    def __init__(self, model: str | None = None):
        import os
        from openai import OpenAI
        from llm import _normalize_base_url
        self.model_name = model or os.getenv("EMB_API_MODEL", "text-embedding-3-small")
        self.client = OpenAI(base_url=_normalize_base_url(config.LLM_BASE_URL), api_key=config.LLM_API_KEY)
        self.dim = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 128):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i:i + 128])
            out.extend(d.embedding for d in resp.data)
        v = np.asarray(out, dtype="float32")
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        v = v / norm
        self.dim = v.shape[1]
        return v

    def embed_documents(self, texts) -> np.ndarray:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]


def get_embedder(backend: str = "local"):
    if backend == "local":
        return LocalBGEEmbedder()
    if backend == "api":
        return ApiEmbedder()
    raise ValueError(f"未知 backend: {backend}（可选 local / api）")
