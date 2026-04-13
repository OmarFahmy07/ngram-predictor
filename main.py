import argparse
import os

import nltk
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer


def main():
    # Load configuration from .env
    load_dotenv("config/.env")

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="N-Gram Next-Word Predictor (Capstone Project)"
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        help="Which pipeline step to run."
    )

    args = parser.parse_args()

    # -----------------------------
    # M1: Data Preparation
    # -----------------------------
    if args.step == "dataprep":
        # Ensure required NLTK data is available
        nltk.download("punkt_tab", quiet=True)

        normalizer = Normalizer()

        train_raw_dir = os.getenv("TRAIN_RAW_DIR")
        train_tokens_path = os.getenv("TRAIN_TOKENS")

        # 1. Load raw text
        text = normalizer.load(train_raw_dir)

        # 2. Strip Project Gutenberg header/footer
        text = normalizer.strip_gutenberg(text)

        # 3. Sentence tokenize
        sentences = normalizer.sentence_tokenize(text)

        # 4. Keep only first 100 sentences (M1 sanity check requirement)
        sentences = sentences[:100]

        # 5. Normalize each sentence and word tokenize
        tokenized_sentences = []
        for sentence in sentences:
            sentence = normalizer.normalize(sentence)

            tokens = normalizer.word_tokenize(sentence)
            tokenized_sentences.append(tokens)

        # 6. Save output
        normalizer.save(tokenized_sentences, train_tokens_path)


if __name__ == "__main__":
    main()