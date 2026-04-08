import os
import re
from typing import List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


class Normalizer:
    """
    Loads, cleans, normalizes, tokenizes, and saves text data.

    Used in:
    - Module 1 (Data Prep): process full corpus
    - Module 3 (Inference): normalize single input strings
    """

    def load(self, folder_path: str) -> str:
        """Load and concatenate all .txt files from a folder."""
        texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                path = os.path.join(folder_path, filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
        return "\n".join(texts)

    def strip_gutenberg(self, text: str) -> str:
        """Remove Project Gutenberg header and footer."""
        start_marker = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
        end_marker = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)

        if start_marker and end_marker:
            return text[start_marker.end():end_marker.start()]

        return text

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", " ", text)

    def remove_numbers(self, text: str) -> str:
        return re.sub(r"\d+", " ", text)

    def remove_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, text: str) -> str:
        """Apply all normalization steps in order."""
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def word_tokenize(self, sentence: str) -> List[str]:
        return word_tokenize(sentence)

    def save(self, sentences: List[List[str]], filepath: str) -> None:
        """Write tokenized sentences to file, one sentence per line."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for tokens in sentences:
                if tokens:
                    f.write(" ".join(tokens) + "\n")


def main() -> None:
    """Local module test (not pipeline)."""
    print("Normalizer module loaded.")


if __name__ == "__main__":
    main()