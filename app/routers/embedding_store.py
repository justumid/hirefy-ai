# services/embedding_store_service.py

import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger("embedding_store")


class EmbeddingStoreService:
    def __init__(self):
        # Store embeddings as: {id: {"vector": np.array, "type": str, "metadata": dict}}
        self.store: Dict[str, Dict] = {}

    def add_embedding(self, item_id: str, item_type: str, vector: List[float], metadata: Optional[Dict] = None):
        """
        Add or update an embedding vector for a given ID.
        """
        if not vector:
            raise ValueError("Empty vector cannot be added.")

        self.store[item_id] = {
            "vector": np.array(vector, dtype=np.float32),
            "type": item_type,
            "metadata": metadata or {}
        }
        logger.info(f"[EmbeddingStore] Added/Updated: {item_id} ({item_type})")

    def delete_embedding(self, item_id: str):
        """
        Delete embedding vector by ID.
        """
        if item_id not in self.store:
            raise ValueError(f"Item '{item_id}' not found.")
        del self.store[item_id]
        logger.info(f"[EmbeddingStore] Deleted: {item_id}")

    def query_similar(self, query_vector: List[float], top_k: int = 5, type_filter: Optional[str] = None) -> List[Dict]:
        """
        Return top_k most similar embeddings based on cosine similarity.
        """
        if not query_vector:
            raise ValueError("Empty query vector.")

        query_vec = np.array(query_vector, dtype=np.float32)
        if np.linalg.norm(query_vec) == 0:
            raise ValueError("Zero vector cannot be used for similarity search.")

        similarities = []

        for item_id, record in self.store.items():
            if type_filter and record["type"] != type_filter:
                continue
            stored_vec = record["vector"]
            sim = self._cosine_similarity(query_vec, stored_vec)
            similarities.append({
                "id": item_id,
                "score": round(sim, 6),
                "metadata": record.get("metadata", {})
            })

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / denom)
