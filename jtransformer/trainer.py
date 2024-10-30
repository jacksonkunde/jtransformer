from jtransformer.config import TrainingConfig
from jtransformer.modules import Jtransformer

from typing import Dict
import os

from transformers import PreTrainedTokenizer
from jaxtyping import Int
import torch as th
from torch.utils.data import DataLoader
from torch import nn
import wandb
from tqdm import tqdm

from datasets import Dataset, load_dataset, load_from_disk


class Jtrainer:
    def __init__(
        self, cfg: TrainingConfig, model: Jtransformer, tokenizer: PreTrainedTokenizer
    ) -> None:
        self.cfg = cfg
        if not os.path.exists(cfg.save_path):
            print(f"Creating save folder {cfg.save_path}")
            os.makedirs(cfg.save_path, exist_ok=True)
        self.device = cfg.device
        self.debug = cfg.debug
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = th.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.n_steps = 0
        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None

    def train_step(self, batch: Dict[str, Int[th.Tensor, "batch seq"]]):
        input_ids = batch["input_ids"].to(self.device)

        # Shift input to create labels: labels[i] = input_ids[i+1]
        labels = input_ids[:, 1:].clone()  # Exclude first token for labels
        input_ids = input_ids[:, :-1]  # Exclude last token for input

        # Forward pass through the model
        logits = self.model(input_ids)  # logits: [batch_size, seq_length-1, vocab_size]

        # Reshape logits and labels for CrossEntropyLoss
        logits = logits.view(
            -1, logits.size(-1)
        )  # [batch_size * (seq_length-1), vocab_size]
        labels = labels.view(-1)  # [batch_size * (seq_length-1)]

        # Compute loss, ignoring padding tokens
        loss = self.criterion(logits, labels)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_steps += 1
        if not self.debug:
            wandb.log({"train_loss": loss}, step=self.n_steps)
        else:
            print(f"Step: {self.n_steps}, Train_loss: {loss}")

        return loss

    def val_step(self, batch: Dict[str, Int[th.Tensor, "batch seq"]]):
        tokens = batch["input_ids"].to(self.device)
        logits = self.model(tokens)[:, :-1]
        predicted_tokens = logits.argmax(dim=-1)
        correct_predictions = (predicted_tokens == tokens[:, 1:]).flatten()
        return correct_predictions

    def train(self):
        # Now load the dataloaders for training and validation
        self._load_train_loader()
        self._load_val_loader()

        if not self.debug:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_display_name,
                config=self.cfg,
            )

        accuracy: float | None = None
        progress_bar = tqdm(total=self.cfg.max_steps_per_epoch * self.cfg.n_epochs)

        for epoch in range(self.cfg.n_epochs):
            for i, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch+1}, Training Loss: {loss:.3f}, Most Recent Validation Accuracy: {accuracy:.3f}"
                    if accuracy
                    else f"Epoch {epoch+1}, Training Loss: {loss:.3f}"
                )
                if i >= self.cfg.max_steps_per_epoch:
                    break

            if epoch > 0 and epoch % self.cfg.save_freq == 0:
                save_dir = os.path.join(self.cfg.save_path, f"epoch_{epoch + 1}")
                self.model.save(save_dir)

            correct_predictions = th.concat(
                [self.val_step(batch) for batch in self.val_dataloader]
            )
            accuracy = correct_predictions.float().mean().item()
            if not self.debug:
                wandb.log({"accuracy": accuracy}, step=self.n_steps)

        print(f"Final Validation Accuracy: {accuracy}")
        save_dir = os.path.join(self.cfg.save_path, "final")
        self.model.save(save_dir)

    def _load_train_loader(self) -> None:
        if self.train_dataloader is None:
            train_dataset = load_from_disk(self.cfg.train_data_path)
            train_dataset.set_format(type="torch", columns=["input_ids"])
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
            )

    def _load_val_loader(self) -> None:
        if self.val_dataloader is None:
            val_dataset = load_from_disk(self.cfg.val_data_path)
            val_dataset.set_format(type="torch", columns=["input_ids"])
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

    @classmethod
    def create_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        file_path: str | None = None,
        hf_dataset_name: str | None = None,
        tokenizer_kwargs: dict = {},
        chunk_size: int | None = None,
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

        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

        return tokenized_dataset
