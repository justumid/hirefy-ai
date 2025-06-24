# models/sentence_bert.py

import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer, util
import numpy as np

logger = logging.getLogger("sentence_bert")


class SentenceBERTEmbedder:
    """
    Wrapper for sentence embedding tasks like:
    - Resume ↔ JD similarity
    - Semantic search over skills
    - Clustering candidates
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            self.name = model_name
            logger.info(f"[SentenceBERT] Loaded model: {model_name}")
        except Exception as e:
            logger.exception(f"[SentenceBERT] Failed to load model: {e}")
            raise RuntimeError(f"Failed to load SentenceBERT model: {e}")

    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Embed a single sentence or a list of sentences into dense vectors.
        """
        if isinstance(texts, str):
            embedding = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            logger.debug(f"[Embed] Single input shape: {embedding.shape}")
            return embedding
        elif isinstance(texts, list):
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            logger.debug(f"[Embed] Batch input size: {len(embeddings)}")
            return embeddings
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def cosine_similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]) -> float:
        """
        Compute average cosine similarity between two strings or batches.
        """
        emb_a = self.embed(a)
        emb_b = self.embed(b)

        if isinstance(a, str) and isinstance(b, str):
            score = float(util.cos_sim(emb_a, emb_b)[0][0])
            logger.debug(f"[Similarity] Cosine: {score:.4f}")
            return score
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            sims = [float(util.cos_sim(va, vb)[0][0]) for va, vb in zip(emb_a, emb_b)]
            avg_score = float(np.mean(sims))
            logger.debug(f"[Similarity] Batch Avg Cosine: {avg_score:.4f}")
            return avg_score
        else:
            raise ValueError("Mismatch in input types or lengths for similarity comparison.")

    def search_top_k(self, query: str, corpus: List[str], top_k: int = 5) -> List[dict]:
        """
        Semantic search to find top-k similar sentences in a corpus.
        """
        try:
            query_emb = self.embed(query)
            corpus_embs = self.embed(corpus)

            scores = util.cos_sim(query_emb, corpus_embs)[0]
            top_results = np.argsort(-scores)[:top_k]

            results = [{
                "text": corpus[i],
                "score": float(scores[i])
            } for i in top_results]

            logger.debug(f"[Search] Top-{top_k} results: {results}")
            return results
        except Exception as e:
            logger.exception(f"[Search Error] {e}")
            return []
