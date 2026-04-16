import argparse
import os

import nltk
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def run_dataprep_100(normalizer: Normalizer) -> None:
    """M1 sample dataprep: produce train_tokens.txt from first 100 sentences."""
    
    # Ensure required NLTK data is available
    nltk.download("punkt_tab", quiet=True)

    # Configurations
    train_raw_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens_path = os.getenv("TRAIN_TOKENS")

    # 1. Load raw text
    text = normalizer.load(train_raw_dir)

    # 2. Strip Project Gutenberg header/footer
    text = normalizer.strip_gutenberg(text)

    # 3. Sentence tokenize
    sentences = normalizer.sentence_tokenize(text)

    # 4. Keep only first 100 sentences (M1 sanity check requirement)
    # sentences = sentences[:100]

    # 5. Normalize each sentence and word tokenize
    tokenized_sentences = []
    for sentence in sentences:
        sentence = normalizer.normalize(sentence)
        tokens = normalizer.word_tokenize(sentence)
        tokenized_sentences.append(tokens)

    # 6. Save output
    normalizer.save(tokenized_sentences, train_tokens_path)

def run_model_build(model: NGramModel) -> None:
    """M2 model build: produce vocab.json and model.json from train_tokens.txt."""
    
    # Configurations
    token_file = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    
    # Build vocab and model (counts + probabilities)
    model.build_vocab(token_file)
    model.build_counts_and_probabilities(token_file)
    model.save_vocab(vocab_path)
    model.save_model(model_path)

def run_inference_loop(predictor: Predictor, k: int) -> None:
    """M3 interactive CLI loop."""
    print("Type text to get predictions. Type 'quit' to exit.")
    try:
        while True:
            text = input("> ").strip()
            if text.lower() == "quit":
                print("Goodbye.")
                break

            preds = predictor.predict_next(text, k)
            print(f"Predictions: {preds}")
    except KeyboardInterrupt:
        print("\nGoodbye.")

def main():
    # Load configuration from .env
    load_dotenv("config/.env", override=True)

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

    # Read common config
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    top_k = int(os.getenv("TOP_K"))
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")

    # Instantiate dependencies ONCE in main() (dependency injection)
    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)

    # -----------------------------
    # M1: Data Preparation
    # -----------------------------
    if args.step == "dataprep":
        run_dataprep_100(normalizer)

    # -----------------------------
    # M2: Model
    # -----------------------------
    if args.step == "model":
        run_model_build(model)

    # -----------------------------
    # M3: Inference
    # -----------------------------
    if args.step == "inference":
        # Load model artifacts once, then run CLI
        model.load(model_path, vocab_path)
        predictor = Predictor(model, normalizer)
        run_inference_loop(predictor, top_k)

    
    # -----------------------------
    # M4: All (dataprep -> model -> inference)
    # -----------------------------
    if args.step == "all":
        run_dataprep_100(normalizer)
        run_model_build(model)
        model.load(model_path, vocab_path)
        predictor = Predictor(model, normalizer)
        run_inference_loop(predictor, top_k)


if __name__ == "__main__":
    main()