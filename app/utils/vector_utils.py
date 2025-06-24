import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("vector_utils")

# Load once at module level
_default_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class EmbeddingEncoder:
    def __init__(self, model_name: str = _default_model_name):
        try:
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"[EmbeddingEncoder] Loaded model: {model_name} (dim={self.dim})")
        except Exception as e:
            logger.exception(f"[EmbeddingEncoder] Failed to load model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        try:
            sim = cosine_similarity([emb1], [emb2])[0][0]
            return round(float(sim), 4)
        except Exception as e:
            logger.warning(f"[SimilarityError] {e}")
            return 0.0


def average_embedding(texts: List[str], encoder: EmbeddingEncoder) -> np.ndarray:
    embeddings = encoder.encode(texts)
    avg = np.mean(embeddings, axis=0)
    return avg / np.linalg.norm(avg)


def best_match_score(query: str, candidates: List[str], encoder: EmbeddingEncoder) -> List[dict]:
    query_emb = encoder.encode(query)[0]
    candidate_embs = encoder.encode(candidates)
    scores = cosine_similarity([query_emb], candidate_embs)[0]

    results = [
        {"text": text, "score": round(float(score), 4)}
        for text, score in zip(candidates, scores)
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)
