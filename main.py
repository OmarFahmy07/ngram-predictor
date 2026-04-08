import argparse
import os

import nltk
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer


def main():
    
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(
        description="N-Gram Next-Word Predictor (Capstone Project)"
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        help="Which pipeline step to run."
    )

    args = parser.parse_args()

    if args.step == "dataprep":
        
        nltk.download("punkt_tab", quiet=True)

        normalizer = Normalizer()

        train_raw_dir = os.getenv("TRAIN_RAW_DIR")
        train_tokens_path = os.getenv("TRAIN_TOKENS")

        text = normalizer.load(train_raw_dir)

        text = normalizer.strip_gutenberg(text)

        text = normalizer.lowercase(text)

        sentences = normalizer.sentence_tokenize(text)

        sentences = sentences[:100]

        tokenized_sentences = []
        for sentence in sentences:
            sentence = normalizer.remove_punctuation(sentence)
            sentence = normalizer.remove_numbers(sentence)
            sentence = normalizer.remove_whitespace(sentence)

            tokens = normalizer.word_tokenize(sentence)
            tokenized_sentences.append(tokens)

        normalizer.save(tokenized_sentences, train_tokens_path)


if __name__ == "__main__":
    main()