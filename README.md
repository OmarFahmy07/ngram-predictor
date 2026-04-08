# N-Gram Next-Word Predictor

A next-word prediction system built from scratch using a statistical n-gram language model.
The model is trained on four *Sherlock Holmes* novels by Arthur Conan Doyle (Project Gutenberg).
At inference time, the system takes the last `NGRAM_ORDER - 1` words typed by the user and predicts the top-k most probable next words using an n-gram model with backoff to lower-order contexts when the input context is unseen.

## Requirements

- Python 3.14.3
- Install all dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/OmarFahmy07/ngram-predictor.git
   cd ngram-predictor
   ```

2. **Create and activate an Anaconda environment**
   ```bash
   conda create -n ngram-predictor python=3.14.3
   conda activate ngram-predictor
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `config/.env`**

   Create a file named `.env` inside the `config/` folder with the following contents:
   ```env
   TRAIN_RAW_DIR=data/raw/train/
   TRAIN_TOKENS=data/processed/train_tokens.txt
   MODEL=data/model/model.json
   VOCAB=data/model/vocab.json
   UNK_THRESHOLD=3
   TOP_K=3
   NGRAM_ORDER=4
   ```

   Note:
   - No file paths or thresholds are hardcoded in the code.

5. **Download the raw training data**

   Download the following Project Gutenberg books and place them in `data/raw/train/`:

   - `1661-0.txt` — *The Adventures of Sherlock Holmes*
     https://www.gutenberg.org/files/1661/1661-0.txt
   - `834-0.txt` — *The Memoirs of Sherlock Holmes*
     https://www.gutenberg.org/files/834/834-0.txt
   - `108.txt` — *The Return of Sherlock Holmes*
     https://www.gutenberg.org/files/108/108.txt
   - `2852-0.txt` — *The Hound of the Baskervilles*
     https://www.gutenberg.org/files/2852/2852-0.txt

   The code automatically loads **all `.txt` files** found in the training folder.

## Usage

All project steps are run through `main.py` using the `--step` argument.

1. **Data preparation**
   ```bash
   python main.py --step dataprep
   ```
   Produces:
   - `data/processed/train_tokens.txt`

2. **Model training**
   ```bash
   python main.py --step model
   ```
   Produces:
   - `data/model/model.json`
   - `data/model/vocab.json`

3. **Inference (interactive CLI)**
   ```bash
   python main.py --step inference
   ```
   Starts an interactive prompt where you input text and receive top-k predictions.
   Exit by typing `quit` or pressing `Ctrl+C`.

4. **Run the full pipeline**
   ```bash
   python main.py --step all
   ```
   Runs:
   ```
   dataprep → model → inference
   ```

## Project Structure

```text
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/          # Four training books (.txt)
│   │   └── eval/           # Evaluation book (.txt) — extra credit only
│   ├── processed/
│   │   ├── train_tokens.txt
│   │   └── eval_tokens.txt # Extra credit only
│   └── model/
│       ├── model.json      # Generated — do not commit
│       └── vocab.json      # Generated — do not commit
├── src/
│   ├── data_prep/
│   │   └── normalizer.py   # Normalizer class
│   ├── model/
│   │   └── ngram_model.py  # NGramModel class
│   └── inference/
│       └── predictor.py    # Predictor class
├── main.py                 # Single entry point — CLI and wiring
├── .gitignore
├── requirements.txt
└── README.md
```
