from jtransformer.config import TrainingConfig
from jtransformer.char_tokenizer import CharTokenizer

import os
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod  # For extensible trainers

from transformers import PreTrainedTokenizer
import torch as th
from torch.utils.data import DataLoader
from torch import nn, optim
import wandb
from tqdm import tqdm

from datasets import Dataset, load_dataset, load_from_disk


class Jtrainer(ABC):
    def __init__(
        self,
        cfg: TrainingConfig,
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, CharTokenizer],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.optimizer: Optional[optim.Optimizer] = self._setup_optimizer()
        self.criterion: Optional[nn.Module] = self._setup_loss()
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.n_steps = 0

    @abstractmethod
    def _setup_loss(self) -> nn.Module:
        """Subclasses implement their specific loss function setup."""
        pass

    @abstractmethod
    def _setup_optimizer(self) -> optim.Optimizer:
        """Subclasses implement their optimizer setup."""
        pass

    @abstractmethod
    def val_metric(self, predictions: th.Tensor, targets: th.Tensor) -> float:
        """Subclasses implement their own validation metric."""
        pass

    def train_step(self, batch: Dict[str, th.Tensor]) -> float:
        """Shared logic for a single training step."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["label"].to(self.device)

        predictions = self.model(input_ids).squeeze(-1)
        loss = self.criterion(predictions, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_steps += 1
        wandb.log({"train_loss": loss.item()}, step=self.n_steps)
        return loss.item()

    def val_step(self, batch: Dict[str, th.Tensor]) -> float:
        """Shared logic for a single validation step."""
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["label"].to(self.device)

        with th.no_grad():
            predictions = self.model(input_ids).squeeze(-1)
            return self.val_metric(predictions, labels)

    def train(self) -> None:
        """Main training loop, shared across trainers."""
        self.model.train()
        # Now load the dataloaders for training and validation
        self._load_data()
        assert (
            self.train_dataloader is not None
        ), "There was a problem loading the data into the training dataloader."
        assert (
            self.val_dataloader is not None
        ), "There was a problem loading the data into the validation dataloader."

        if not self.cfg.debug:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_display_name,
                config=self.cfg.to_dict(),
            )

        progress_bar = tqdm(total=self.cfg.n_epochs * self.cfg.max_steps_per_epoch)

        for epoch in range(self.cfg.n_epochs):
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss:.4f}")

            if epoch % self.cfg.save_freq == 0:
                self.save_model(f"epoch_{epoch+1}")

            val_loss = sum(self.val_step(b) for b in self.val_dataloader) / len(
                self.val_dataloader
            )
            wandb.log({"val_loss": val_loss}, step=self.n_steps)

        self.save_model("final")

    def save_model(self, save_name: str) -> None:
        """Save the model with error handling."""
        try:
            save_path = os.path.join(self.cfg.save_path, save_name)
            os.makedirs(save_path, exist_ok=True)
            th.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def _load_data(self) -> None:
        """Load train and validation datasets."""
        if self.train_dataloader is None:
            train_dataset = load_from_disk(self.cfg.train_data_path)
            train_dataset.set_format(type="torch", columns=["input_ids"])
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_workers,
                shuffle=True,
            )
        if self.val_dataloader is None:
            val_dataset = load_from_disk(self.cfg.val_data_path)
            val_dataset.set_format(type="torch", columns=["input_ids"])
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_workers,
                shuffle=False,
            )

    @classmethod
    def create_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        file_path: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        tokenizer_kwargs: dict = {},
        chunk_size: Optional[int] = None,
        overlap_size: int = 0,
    ) -> Dataset:
        """
        Converts a plain text file into a tokenized dataset for next-token prediction.
        """

        def read_txt_to_dict_chunks(
            file_path: str,
            tokenizer: PreTrainedTokenizer,
            chunk_size: int,
            overlap_size: int,
            tokenizer_kwargs: dict = {},
        ) -> list[dict]:
            """
            Streams a large text file in chunks with overlap to avoid loss of context.
            Each chunk contains `chunk_size` tokens, with `overlap_size` tokens overlapping between chunks.
            """
            chunks = []
            current_chunk = []

            # Open the file and read it line by line
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Ignore empty lines
                    line = line.strip()
                    if not line:
                        continue

                    # Tokenize the line and append to the current chunk
                    tokens = tokenizer(line + "\n", padding=False, truncation=False)[
                        "input_ids"
                    ]
                    current_chunk.extend(tokens)

                    # If the current chunk exceeds the desired chunk size, save it
                    while len(current_chunk) >= chunk_size:
                        chunks.append({"input_ids": current_chunk[:chunk_size]})

                        # Create the next chunk starting with the overlapping part
                        current_chunk = current_chunk[chunk_size - overlap_size :]

            # Add the last chunk if any tokens remain
            if current_chunk:
                if len(current_chunk) < chunk_size:
                    current_chunk.extend(
                        [tokenizer.pad_token_id] * (chunk_size - len(current_chunk))
                    )
                chunks.append({"input_ids": current_chunk})

            return chunks

        def read_txt_to_dict_lines(file_path: str):
            """Reads a plain text file and returns a list of dictionaries containing plain text."""
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [{"text": line.strip() + "\n"} for line in lines if line.strip()]

        def tokenize_function(examples):
            """Tokenizes the input text using the provided tokenizer."""
            return tokenizer(examples["text"], **tokenizer_kwargs)

        if hf_dataset_name:
            tokenized_dataset = load_dataset(hf_dataset_name).map(
                tokenize_function, batched=True
            )
        elif file_path:
            # Read text and convert to dataset
            if chunk_size:
                data_dict = read_txt_to_dict_chunks(
                    file_path=file_path,
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                tokenized_dataset = Dataset.from_list(data_dict)
            else:
                data_dict = read_txt_to_dict_lines(file_path)
                tokenized_dataset = Dataset.from_list(data_dict).map(
                    tokenize_function, batched=True
                )
        else:
            raise Exception

        return tokenized_dataset


class NextTokenPredictionTrainer(Jtrainer):
    def _setup_loss(self) -> nn.Module:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", -100)
        return nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def _setup_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def val_metric(self, predictions: th.Tensor, targets: th.Tensor) -> float:
        correct = (predictions.argmax(dim=-1) == targets).float().mean().item()
        return correct

    def train_step(self, batch: Dict[str, th.Tensor]) -> float:
        """Override to handle input-label shifting."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)

        # Shift inputs to create labels for next-token prediction
        labels = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]

        # Forward pass and loss calculation
        logits = self.model(input_ids)  # [batch_size, seq_len-1, vocab_size]
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_steps += 1
        wandb.log({"train_loss": loss.item()}, step=self.n_steps)
        return loss.item()
