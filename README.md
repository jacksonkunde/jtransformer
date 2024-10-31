# jtransformer

**jtransformer** is a lightweight transformer library built from scratch in PyTorch, following the architecture of GPT-2. This project serves as both an educational tool for understanding the internals of transformer models and a starting point for experimentation with transformers.

---

## Features

- **From-Scratch GPT-2 Implementation:** The transformer architecture and components, including embeddings, attention, and MLP layers, are implemented without relying on pre-built models.
- **Training Support:** Includes a `Jtrainer` class for next-token prediction training.
- **Configuration Management:** Uses structured configs for model and training settings (`config.py`).
- **Character-Level Tokenizer:** Extendable support for character-level tokenization.
- **Save and Load Model States:** Easily save and reload models and configurations to/from disk.
- **Shakespeare Example:** A ready-to-use example for training a character-level transformer.

---

## Installation

### Install the jtransformer Package Via `pip`
```bash
pip install git+https://github.com/jacksonkunde/jtransformer.git
```

### Or Clone the Repository and Install Dependencies
```bash
git clone https://github.com/jacksonkunde/jtransformer.git
cd jtransformer
pip install -r requirements.txt
```
---

## Usage

### Train the Model on Shakespeare Data
Download the sample dataset:
```bash
wget -P cookbook/shakespeare/ https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

#### You can use the `Jtrainer` class to start training. 
- Use the example in [cookbook/shakespeare/train_shakespeare.py](cookbook/shakespeare/train_shakespeare.py)
- Or un this notebook on **Google Colab** [HERE :smile:](https://colab.research.google.com/drive/19QjORQJRrurgmpe9jdNTZWmZW1VuyGy3?usp=sharing)

### Run Tests
We provide `pytest` tests for core components inside `test_modules.py`. Run the following command to execute tests:
```bash
.venv/bin/python -m pytest -v test_modules.py
```

---

## Project Structure

```
jtransformer/
│
├── char_tokenizer.py      # Custom tokenizer implementation
├── config.py              # Configuration classes for models and training
├── modules.py             # Core transformer components and architecture
├── trainer.py             # Training logic for next-token prediction
└── test_modules.py        # Pytest unit tests for core modules
```

---

## To-Do List
- [ ] Add mixed-precision training and gradient accumulation support
- [ ] Increase compatibility with HuggingFace libraries
- [ ] Enforce stricter type checking
- [ ] Extend the character tokenizer to inherit from HuggingFace's `PreTrainedTokenizer`
- [ ] Add sampling methods (top_p, top_k, etc.)

---

## Acknowledgements
- [Andrej Karpathy](https://github.com/karpathy) for the Shakespeare dataset and inspiration from nanoGPT.
- [ARENA](https://www.arena.education/) for inspiration and guidence.
---