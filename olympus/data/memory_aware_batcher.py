"""
Memory-aware batching that clusters related documents together.

Uses simple TF-IDF-like similarity to group documents into coherent
batches, falling back to random batching when similarity data is
unavailable or documents are too few.
"""

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import logging

logger = logging.getLogger(__name__)


class MemoryAwareBatcher:
    """Clusters related documents into batches using TF-IDF similarity.

    The batcher computes lightweight TF-IDF vectors for each document,
    then greedily assigns documents to batches by picking the most
    similar remaining document to the current batch centroid.

    Args:
        max_vocab: Maximum vocabulary size for the TF-IDF representation.
        min_docs_for_clustering: If fewer documents are provided,
                                  fall back to random batching.
        similarity_threshold: Minimum cosine similarity to add a document
                              to the current cluster.  Below this, start
                              a new batch.
    """

    def __init__(
        self,
        max_vocab: int = 10_000,
        min_docs_for_clustering: int = 8,
        similarity_threshold: float = 0.05,
    ) -> None:
        self.max_vocab = max_vocab
        self.min_docs_for_clustering = min_docs_for_clustering
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # TF-IDF helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenizer."""
        return text.lower().split()

    def _build_tfidf(
        self, documents: List[str]
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """Compute TF-IDF vectors for each document.

        Returns:
            - List of TF-IDF dicts (one per document).
            - IDF dict mapping term -> idf value.
        """
        n_docs = len(documents)

        # Document frequency
        df: Counter = Counter()
        doc_tokens: List[List[str]] = []
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                df[tok] += 1

        # Keep only the top-k terms by document frequency
        if len(df) > self.max_vocab:
            top_terms = {t for t, _ in df.most_common(self.max_vocab)}
        else:
            top_terms = set(df.keys())

        # IDF
        idf: Dict[str, float] = {}
        for term in top_terms:
            idf[term] = math.log((n_docs + 1) / (df[term] + 1)) + 1.0

        # TF-IDF per document
        tfidf_vectors: List[Dict[str, float]] = []
        for tokens in doc_tokens:
            tf: Counter = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec: Dict[str, float] = {}
            for term, count in tf.items():
                if term in idf:
                    vec[term] = (count / total) * idf[term]
            # L2 normalise
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vec = {k: v / norm for k, v in vec.items()}
            tfidf_vectors.append(vec)

        return tfidf_vectors, idf

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        if not a or not b:
            return 0.0
        # a and b are already L2-normalised, so dot product = cosine
        dot = sum(a[k] * b[k] for k in a if k in b)
        return dot

    @staticmethod
    def _centroid(vectors: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute the (normalised) centroid of a list of sparse vectors."""
        if not vectors:
            return {}
        merged: Dict[str, float] = defaultdict(float)
        for v in vectors:
            for k, val in v.items():
                merged[k] += val
        n = len(vectors)
        centroid = {k: val / n for k, val in merged.items()}
        norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
        return {k: v / norm for k, v in centroid.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_batches(
        self,
        documents: List[str],
        batch_size: int,
    ) -> List[List[str]]:
        """Create batches of related documents.

        Args:
            documents: List of document texts.
            batch_size: Target batch size.

        Returns:
            List of batches, where each batch is a list of document strings.
        """
        if not documents:
            return []

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Fall back to random batching for small sets
        if len(documents) < self.min_docs_for_clustering:
            return self._random_batches(documents, batch_size)

        try:
            return self._clustered_batches(documents, batch_size)
        except Exception as e:
            logger.warning("Clustering failed (%s), falling back to random batching", e)
            return self._random_batches(documents, batch_size)

    def _random_batches(
        self, documents: List[str], batch_size: int
    ) -> List[List[str]]:
        """Simple random batching."""
        docs = list(documents)
        random.shuffle(docs)
        batches: List[List[str]] = []
        for i in range(0, len(docs), batch_size):
            batches.append(docs[i : i + batch_size])
        return batches

    def _clustered_batches(
        self, documents: List[str], batch_size: int
    ) -> List[List[str]]:
        """Greedy clustering-based batching."""
        tfidf_vectors, _ = self._build_tfidf(documents)

        remaining: Set[int] = set(range(len(documents)))
        batches: List[List[str]] = []

        while remaining:
            # Start a new batch with a random seed document
            seed = random.choice(list(remaining))
            remaining.remove(seed)

            batch_indices = [seed]
            batch_vectors = [tfidf_vectors[seed]]

            while len(batch_indices) < batch_size and remaining:
                centroid = self._centroid(batch_vectors)
                # Find the most similar remaining document
                best_idx = -1
                best_sim = -1.0
                for idx in remaining:
                    sim = self._cosine_similarity(centroid, tfidf_vectors[idx])
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx

                # If similarity is too low, stop this batch
                if best_sim < self.similarity_threshold:
                    break

                remaining.remove(best_idx)
                batch_indices.append(best_idx)
                batch_vectors.append(tfidf_vectors[best_idx])

            batches.append([documents[i] for i in batch_indices])

        return batches
