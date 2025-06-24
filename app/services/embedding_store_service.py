import os
import faiss
import json
import threading
import hashlib
import numpy as np
import logging
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from app.base.models import EmbeddingRecord, EmbeddingSearchResult

logger = logging.getLogger("embedding_store")

# === Constants ===
EMBEDDING_DIM = 384  # Compatible with all-MiniLM-L6-v2
INDEX_PATH = os.getenv("EMBEDDING_INDEX_PATH", "data/faiss.index")
META_PATH = os.getenv("EMBEDDING_META_PATH", "data/embedding_metadata.json")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingStoreService:
    def __init__(self, model_name: Optional[str] = None):
        self.model = SentenceTransformer(model_name or MODEL_NAME)
        self.index = self._load_index()
        self.metadata: Dict[int, EmbeddingRecord] = self._load_metadata()
        self.cache: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.id_counter = max(self.metadata.keys(), default=-1) + 1

    def _load_index(self):
        if os.path.exists(INDEX_PATH):
            logger.info("[EmbeddingStore] Loading existing FAISS index")
            return faiss.read_index(INDEX_PATH)
        logger.info("[EmbeddingStore] Creating new FAISS index")
        return faiss.IndexFlatIP(EMBEDDING_DIM)

    def _load_metadata(self) -> Dict[int, EmbeddingRecord]:
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                    return {int(k): EmbeddingRecord(**v) for k, v in raw_meta.items()}
            except Exception as e:
                logger.warning(f"[EmbeddingStore] Metadata load failed: {e}")
        return {}

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _encode_text(self, text: str) -> np.ndarray:
        h = self._hash_text(text)
        if h in self.cache:
            return self.cache[h]

        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        if emb.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"Embedding shape mismatch: expected {EMBEDDING_DIM}, got {emb.shape[0]}")

        self.cache[h] = emb
        return emb

    def add_embedding(self, record: EmbeddingRecord) -> int:
        with self.lock:
            try:
                emb_vec = self._encode_text(record.text)
                self.index.add(np.array([emb_vec]))
                current_id = self.id_counter
                self.metadata[current_id] = record
                self.id_counter += 1
                self._persist()
                logger.info(f"[EmbeddingStore] Added id={current_id}, type={record.type}")
                return current_id
            except Exception as e:
                logger.exception(f"[EmbeddingStore] Failed to add: {e}")
                raise RuntimeError("Failed to add embedding")

    def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        type_filter: Optional[str] = None
    ) -> List[EmbeddingSearchResult]:
        if self.index.ntotal == 0:
            logger.warning("[EmbeddingStore] No embeddings in index")
            return []

        query_vec = self._encode_text(query_text)
        D, I = self.index.search(np.array([query_vec]), top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx not in self.metadata:
                continue
            record = self.metadata[idx]
            if type_filter and record.type != type_filter:
                continue
            results.append(EmbeddingSearchResult(
                id=idx,
                score=float(score),
                payload=record.dict()
            ))
        return results

    def _persist(self):
        try:
            faiss.write_index(self.index, INDEX_PATH)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump({k: v.dict() for k, v in self.metadata.items()}, f, indent=2)
            logger.info("[EmbeddingStore] Index + metadata persisted")
        except Exception as e:
            logger.warning(f"[EmbeddingStore] Persist failed: {e}")

    def reset_store(self):
        with self.lock:
            self.index.reset()
            self.metadata.clear()
            self.cache.clear()
            self.id_counter = 0
            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)
            if os.path.exists(META_PATH):
                os.remove(META_PATH)
            logger.warning("[EmbeddingStore] Store reset complete")
