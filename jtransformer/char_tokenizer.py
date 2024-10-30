import json
from typing import List

import torch as th


# TODO: Inherit from PreTrainedTokenizer
class CharTokenizer:
    def __init__(self, vocab=None, pad_token_id=0, eos_token_id=1):
        # Initialize the vocabulary with the pad token
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab = vocab or {"<pad>": pad_token_id, "<eos>": eos_token_id}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = max(self.vocab.values()) + 1  # Start from the next available ID

    def _get_token_id(self, char):
        """Assign a unique ID to each character."""
        if char not in self.vocab:
            # Assign a new ID to unseen characters
            self.vocab[char] = self.next_id
            self.inverse_vocab[self.next_id] = char
            self.next_id += 1
        return self.vocab[char]

    def __call__(
        self, text, max_length=None, padding=False, truncation=False, **kwargs
    ):
        """Tokenize the text at the character level and return input IDs."""
        batch_input_ids = []

        if isinstance(text, list):
            for t in text:
                input_ids = self._tokenize(t, max_length, truncation, padding)
                batch_input_ids.append(input_ids)

        else:
            batch_input_ids = self._tokenize(text, max_length)

        return {"input_ids": batch_input_ids}

    def _tokenize(
        self,
        text: str,
        max_length: int | None = None,
        truncation: bool = False,
        padding: bool = False,
    ):
        input_ids = [self._get_token_id(char) for char in text]
        if truncation:
            assert (
                max_length is not None
            ), "You must pass a maximum length for both truncation and padding"
            input_ids = self._truncate(input_ids, max_length)
        if padding:
            assert (
                max_length is not None
            ), "You must pass a maximum length for both truncation and padding"
            input_ids = self._pad(input_ids, max_length)
        return input_ids

    def _pad(self, input_ids, max_length):
        return [self.pad_token_id] * (max_length - len(input_ids)) + input_ids

    def _truncate(self, input_ids, max_length):
        if len(input_ids) > max_length:
            return input_ids[:max_length]  # Truncate if longer
        return input_ids

    def decode(self, input_ids: th.Tensor | List[int]):
        """Convert input IDs back to the original string."""
        if isinstance(input_ids, th.Tensor):
            input_ids = input_ids.squeeze()
            input_ids = input_ids.tolist()
        return "".join(
            [self.inverse_vocab[id] for id in input_ids if id != self.pad_token_id]
        )

    def save(self, save_path: str):
        """Save the tokenizer's vocabulary to a JSON file."""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"vocab": self.vocab, "pad_token_id": self.pad_token_id}, f, indent=4
            )

    @classmethod
    def load(cls, load_path: str):
        """Load the tokenizer's vocabulary from a JSON file."""
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], pad_token_id=data["pad_token_id"])
