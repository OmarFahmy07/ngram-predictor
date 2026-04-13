import json
from collections import Counter, defaultdict
from typing import Dict, List


class NGramModel:
    """
    Builds, stores, saves, loads, and queries n-gram probability tables
    using MLE probabilities with backoff from NGRAM_ORDER down to 1-gram.

    Responsibilities (Module 2):
    - build vocabulary with UNK thresholding
    - build n-gram counts for all orders 1..NGRAM_ORDER
    - compute MLE probabilities for all orders 1..NGRAM_ORDER
    - provide a single backoff lookup(context) used by Predictor later
    - save model.json and vocab.json
    """

    def __init__(self, ngram_order: int, unk_threshold: int) -> None:
        """
        Args:
            ngram_order: maximum n-gram order (e.g., 4)
            unk_threshold: words with frequency < unk_threshold become <UNK>
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold

        self.vocab: List[str] = []
        self._vocab_set: set[str] = set()

        # Probability tables:
        # - "1gram": {word: prob}
        # - "2gram": {"w1": {next: prob, ...}}
        # - "3gram": {"w1 w2": {next: prob, ...}}
        self.tables: Dict[str, Dict] = {}

    # -------------------------
    # Helpers
    # -------------------------
    def _read_token_file(self, token_file: str) -> List[List[str]]:
        """Read train_tokens.txt: one sentence per line, tokens separated by spaces."""
        sentences: List[List[str]] = []
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line.split())
        return sentences

    def _map_token(self, w: str) -> str:
        """Map OOV words to <UNK>."""
        return w if w in self._vocab_set else "<UNK>"

    # -------------------------
    # Required API (per spec)
    # -------------------------
    def build_vocab(self, token_file: str) -> None:
        """
        Build vocabulary from token file and apply UNK thresholding.
        Words with count < UNK_THRESHOLD are replaced by <UNK> (not kept in vocab).
        """
        sentences = self._read_token_file(token_file)
        counts = Counter(w for sent in sentences for w in sent)

        vocab = [w for w, c in counts.items() if c >= self.unk_threshold]
        if "<UNK>" not in vocab:
            vocab.append("<UNK>")

        self.vocab = sorted(vocab)
        self._vocab_set = set(self.vocab)

    def build_counts_and_probabilities(self, token_file: str) -> None:
        """
        Count all n-grams at orders 1..NGRAM_ORDER and compute MLE probabilities.
        Probabilities are computed per context (sum to ~1 inside each context).
        """
        if not self.vocab:
            raise ValueError("Vocabulary is empty. Call build_vocab() first.")

        sentences = self._read_token_file(token_file)

        # Map tokens to vocab/<UNK>
        mapped_sentences = []
        for sent in sentences:
            mapped_sentences.append([self._map_token(w) for w in sent])

        # Counts: order -> context_str -> Counter(next_word)
        counts_by_order: Dict[int, Dict[str, Counter]] = {
            order: defaultdict(Counter) for order in range(1, self.ngram_order + 1)
        }

        # Build counts
        for sent in mapped_sentences:
            L = len(sent)
            for order in range(1, self.ngram_order + 1):
                if L < order:
                    continue
                for i in range(L - order + 1):
                    if order == 1:
                        context = ""  # unigram context
                        nxt = sent[i]
                    else:
                        context_words = sent[i : i + order - 1]
                        nxt = sent[i + order - 1]
                        context = " ".join(context_words)
                    counts_by_order[order][context][nxt] += 1

        # Compute probabilities (MLE)
        tables: Dict[str, Dict] = {}

        # 1-gram: flatten "" context
        unigram_counts = counts_by_order[1][""]
        total = sum(unigram_counts.values())
        tables["1gram"] = {w: c / total for w, c in unigram_counts.items()} if total > 0 else {}

        # n>1: per-context distributions
        for order in range(2, self.ngram_order + 1):
            key = f"{order}gram"
            tables[key] = {}
            for ctx, ctr in counts_by_order[order].items():
                denom = sum(ctr.values())
                if denom == 0:
                    continue
                tables[key][ctx] = {w: c / denom for w, c in ctr.items()}

        self.tables = tables

    def lookup(self, context: List[str]) -> Dict[str, float]:
        """
        Backoff lookup:
        Try NGRAM_ORDER down to 1-gram. Returns {word: prob} from the first order
        where the context exists. Returns {} if no match exists at any order.
        """
        if not self.tables:
            return {}

        # Map OOV in context to <UNK>
        ctx = [self._map_token(w) for w in context]

        for order in range(self.ngram_order, 0, -1):
            if order == 1:
                return self.tables.get("1gram", {})

            need = order - 1
            if len(ctx) < need:
                continue

            ctx_key = " ".join(ctx[-need:])
            dist = self.tables.get(f"{order}gram", {}).get(ctx_key, {})
            if dist:
                return dist

        return {}

    def save_model(self, model_path: str) -> None:
        """Save all probability tables to model.json."""
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.tables, f, indent=2, ensure_ascii=False)

    def save_vocab(self, vocab_path: str) -> None:
        """Save vocabulary list to vocab.json."""
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    def load(self, model_path: str, vocab_path: str) -> None:
        """Load model.json and vocab.json into this instance."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self._vocab_set = set(self.vocab)

        with open(model_path, "r", encoding="utf-8") as f:
            self.tables = json.load(f)


def main() -> None:
    """Module-level quick check."""
    print("NGramModel module loaded.")


if __name__ == "__main__":
    main()