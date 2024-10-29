from trainer import create_dataset, TransformerTrainer
from config import TrainingConfig, TransformerConfig
from modules import Transformer
from transformers import AutoTokenizer
from char_tokenizer import CharTokenizer
import os


def main():
    file_path = "/Users/newemployee/Desktop/transformer/cookbook/shakespeare/input.txt"

    tokenizer = CharTokenizer()

    tokenizer_kwargs = {"max_length": 64, "padding": True, "truncation": True}
    dataset = create_dataset(
        tokenizer=tokenizer, file_path=file_path, tokenizer_kwargs=tokenizer_kwargs
    )

    train_dataset = dataset[: int(len(dataset) * 0.9)]
    val_dataset = dataset[int(len(dataset) * 0.9) :]

    print(type(val_dataset))

    dataset.from_dict(train_dataset).save_to_disk("./train_dataset")
    dataset.from_dict(val_dataset).save_to_disk("./val_dataset")

    training_cfg = TrainingConfig(
        debug=False,
        batch_size=32,
        n_epochs=5,
        train_data_path=os.path.abspath("./train_dataset"),
        val_data_path=os.path.abspath("./val_dataset"),
    )

    model_cfg = TransformerConfig(
        d_model=256,
        debug=True,
        n_ctx=64,
        d_mlp=256,
        n_heads=4,
        d_vocab=len(tokenizer.vocab),
    )

    model = Transformer(model_cfg)

    trainer = TransformerTrainer(cfg=training_cfg, model=model, tokenizer=tokenizer)

    trainer.train()


if __name__ == "__main__":
    main()
