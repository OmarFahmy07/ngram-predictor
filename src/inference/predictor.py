from typing import List

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


class Predictor:
    """
    Accepts a pre-loaded NGramModel and a Normalizer via dependency injection.
    Normalizes user input, extracts the last (NGRAM_ORDER - 1) words as context,
    maps OOV words to <UNK>, calls NGramModel.lookup() (backoff lives only there),
    and returns top-k next-word predictions sorted by probability.
    """

    def __init__(self, model: NGramModel, normalizer: Normalizer) -> None:
        """
        Args:
            model: Pre-loaded NGramModel instance. Do not load files here.
            normalizer: Normalizer instance. Do not re-implement normalization here.
        """
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text: str) -> List[str]:
        """
        Normalize input text using Normalizer.normalize(), then extract context tokens:
        last (NGRAM_ORDER - 1) tokens.

        Args:
            text: Raw user input string.

        Returns:
            context: List of context tokens (length <= NGRAM_ORDER - 1).
        """
        cleaned = self.normalizer.normalize(text)
        tokens = cleaned.split() if cleaned else []
        need = max(self.model.ngram_order - 1, 0)
        return tokens[-need:] if need > 0 else []

    def map_oov(self, context: List[str]) -> List[str]:
        """
        Replace out-of-vocabulary words with <UNK>.

        Args:
            context: Context tokens.

        Returns:
            Mapped context tokens.
        """
        vocab_set = set(self.model.vocab)
        return [w if w in vocab_set else "<UNK>" for w in context]

    def predict_next(self, text: str, k: int) -> List[str]:
        """
        Orchestrate: normalize -> map_oov -> model.lookup() -> rank -> top-k.

        Args:
            text: Raw user input string.
            k: Number of predictions to return.

        Returns:
            List of predicted next words (strings), sorted by probability descending.
            Returns [] if no predictions are found.
        """
        if text is None:
            return []

        context = self.normalize(text)
        context = self.map_oov(context)

        dist = self.model.lookup(context)  # backoff is inside lookup()
        if not dist:
            return []

        ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _p in ranked[:k]]


def main() -> None:
    """Local module test stub (module runnable in isolation)."""
    print("Predictor module loaded.")


if __name__ == "__main__":
    main()