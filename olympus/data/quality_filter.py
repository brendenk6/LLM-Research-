"""
Quality filtering for training data using heuristic scoring.

Provides a simple, fast quality filter that scores text based on
surface-level features without requiring any ML model.
"""

import math
import string
from typing import Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class QualityFilter:
    """Heuristic quality filter for training text.

    Scores text on a 0-to-1 scale based on multiple surface features:

    - Average word length (penalises very short or very long words)
    - Unique word ratio (type-token ratio)
    - Punctuation presence
    - Line length variance (too uniform = boilerplate, too wild = noise)
    - Special character ratio (high ratio = code dump or garbage)

    Args:
        min_words: Minimum number of words to consider text valid.
        weights: Optional dict mapping feature names to their weights.
                 Defaults give equal weight to each feature.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "avg_word_length": 0.20,
        "unique_word_ratio": 0.25,
        "has_punctuation": 0.15,
        "line_length_variance": 0.15,
        "special_char_ratio": 0.25,
    }

    def __init__(
        self,
        min_words: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.min_words = min_words
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # Normalise weights so they sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Feature scorers (each returns a float in [0, 1])
    # ------------------------------------------------------------------

    @staticmethod
    def _score_avg_word_length(words: List[str]) -> float:
        """Ideal average word length is around 4-7 characters."""
        if not words:
            return 0.0
        avg = sum(len(w) for w in words) / len(words)
        # Bell curve centred at 5.5, sigma ~2.5
        return math.exp(-0.5 * ((avg - 5.5) / 2.5) ** 2)

    @staticmethod
    def _score_unique_word_ratio(words: List[str]) -> float:
        """Type-token ratio. Very low = repetitive, very high (for long texts) = unusual."""
        if not words:
            return 0.0
        ratio = len(set(w.lower() for w in words)) / len(words)
        # Ideal range roughly 0.3 - 0.8; penalise extremes gently
        if ratio < 0.1:
            return ratio / 0.1 * 0.3
        if ratio > 0.95:
            return max(0.5, 1.0 - (ratio - 0.95) / 0.05)
        return min(1.0, ratio / 0.6)

    @staticmethod
    def _score_has_punctuation(text: str) -> float:
        """Presence of sentence-ending punctuation and commas."""
        sentence_enders = sum(1 for c in text if c in ".!?")
        commas = text.count(",")
        words = len(text.split())
        if words == 0:
            return 0.0
        # Expect roughly 1 sentence-ender per 15 words and 1 comma per 20 words
        ender_ratio = min(1.0, sentence_enders / max(1, words / 15))
        comma_ratio = min(1.0, commas / max(1, words / 20))
        return 0.7 * ender_ratio + 0.3 * comma_ratio

    @staticmethod
    def _score_line_length_variance(text: str) -> float:
        """Lines with moderate variance are better than perfectly uniform or chaotic."""
        lines = text.split("\n")
        if len(lines) < 2:
            # Single-line text: neutral score
            return 0.6
        lengths = [len(line) for line in lines if line.strip()]
        if not lengths:
            return 0.0
        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0:
            return 0.0
        variance = sum((ln - mean_len) ** 2 for ln in lengths) / len(lengths)
        cv = math.sqrt(variance) / (mean_len + 1e-8)  # coefficient of variation
        # Ideal CV around 0.3 - 0.7
        if cv < 0.05:
            return 0.4  # too uniform (boilerplate / tables)
        if cv > 2.0:
            return 0.3  # too chaotic
        return min(1.0, 0.5 + 0.5 * min(cv, 1.0))

    @staticmethod
    def _score_special_char_ratio(text: str) -> float:
        """High ratio of non-alphanumeric, non-whitespace, non-punctuation chars is bad."""
        if not text:
            return 0.0
        allowed = set(string.ascii_letters + string.digits + string.whitespace + string.punctuation)
        special_count = sum(1 for c in text if c not in allowed)
        ratio = special_count / len(text)
        # Low ratio is good
        if ratio < 0.01:
            return 1.0
        if ratio < 0.05:
            return 0.8
        if ratio < 0.15:
            return 0.5
        return max(0.0, 1.0 - ratio)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Compute a heuristic quality score for *text*.

        Args:
            text: The input text to evaluate.

        Returns:
            Float in [0, 1] where higher means better quality.
        """
        words = text.split()

        # Very short texts get a low score automatically
        if len(words) < self.min_words:
            return max(0.0, len(words) / self.min_words * 0.3)

        feature_scores: Dict[str, float] = {
            "avg_word_length": self._score_avg_word_length(words),
            "unique_word_ratio": self._score_unique_word_ratio(words),
            "has_punctuation": self._score_has_punctuation(text),
            "line_length_variance": self._score_line_length_variance(text),
            "special_char_ratio": self._score_special_char_ratio(text),
        }

        # Weighted sum
        total = 0.0
        for name, w in self.weights.items():
            total += w * feature_scores.get(name, 0.0)

        return max(0.0, min(1.0, total))

    def score_detailed(self, text: str) -> Dict[str, float]:
        """Return per-feature scores and the aggregate.

        Returns:
            Dict with keys for each feature plus ``"total"``.
        """
        words = text.split()
        result: Dict[str, float] = {
            "avg_word_length": self._score_avg_word_length(words),
            "unique_word_ratio": self._score_unique_word_ratio(words),
            "has_punctuation": self._score_has_punctuation(text),
            "line_length_variance": self._score_line_length_variance(text),
            "special_char_ratio": self._score_special_char_ratio(text),
        }
        total = sum(self.weights.get(k, 0.0) * v for k, v in result.items())
        result["total"] = max(0.0, min(1.0, total))
        return result

    def filter(self, texts: List[str], threshold: float = 0.7) -> List[str]:
        """Return texts that score above *threshold*.

        Args:
            texts: List of input texts.
            threshold: Minimum quality score to keep.

        Returns:
            Filtered list of texts (order preserved).
        """
        kept: List[str] = []
        for text in texts:
            s = self.score(text)
            if s >= threshold:
                kept.append(text)
        logger.debug("QualityFilter: kept %d / %d texts (threshold=%.2f)",
                      len(kept), len(texts), threshold)
        return kept

    def filter_with_scores(
        self, texts: List[str], threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Like :meth:`filter` but also returns the score for each kept text."""
        results: List[Tuple[str, float]] = []
        for text in texts:
            s = self.score(text)
            if s >= threshold:
                results.append((text, s))
        return results
