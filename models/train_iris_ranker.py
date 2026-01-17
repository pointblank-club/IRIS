import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import argparse


CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 256,
    "feature_mlp_layers": [128, 128],
    "lr": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "target_metric": "runtime",
    "max_seq_len": 64,
    "val_split": 0.2,
    "dropout": 0.1,
}


def parse_features(feature_dict):
    """Converts a dictionary of features into a dictionary of floats."""
    return {k: float(v) for k, v in feature_dict.items()}


class PassSequenceDataset(Dataset):
    """
    Dataset for the ranking model.
    Each sample consists of program features, a pass sequence, and the corresponding metric.
    """

    def __init__(self, data, target_metric, max_seq_len):
        super().__init__()
        self.target_metric = target_metric
        self.max_seq_len = max_seq_len

        self.samples = []
        self.pass_vocab = {"<pad>": 0, "<unk>": 1}
        self.feature_keys = []

        self._process_data(data)
        self._build_vocab()
        self._tokenize_sequences()

        self.feature_scaler = StandardScaler()
        self._normalize_features()

    def _process_data(self, data):
        """Expands the JSON data into individual samples."""
        if not data:
            return

        first_features = parse_features(data[0]["program_features"])
        self.feature_keys = sorted(first_features.keys())

        for entry in data:
            features = parse_features(entry["program_features"])

            ordered_features = [features.get(k, 0.0) for k in self.feature_keys]

            self.samples.append(
                {
                    "features": np.array(ordered_features, dtype=np.float32),
                    "sequence": entry["pass_sequence"],
                    "metric": entry[self.target_metric],
                }
            )

    def _build_vocab(self):
        """Builds a vocabulary of all unique compiler passes."""
        pass_idx = len(self.pass_vocab)
        for sample in self.samples:
            for p in sample["sequence"]:
                if p not in self.pass_vocab:
                    self.pass_vocab[p] = pass_idx
                    pass_idx += 1
        self.vocab_size = len(self.pass_vocab)

    def _tokenize_sequences(self):
        """Converts pass sequences to token IDs with padding/truncation."""
        for sample in self.samples:
            seq = sample["sequence"]
            token_ids = [self.pass_vocab.get(p, self.pass_vocab["<unk>"]) for p in seq]

            if len(token_ids) < self.max_seq_len:
                token_ids.extend(
                    [self.pass_vocab["<pad>"]] * (self.max_seq_len - len(token_ids))
                )
            else:
                token_ids = token_ids[: self.max_seq_len]

            sample["sequence_tokens"] = torch.tensor(token_ids, dtype=torch.long)

    def _normalize_features(self):
        """Fits and transforms program features using StandardScaler."""
        if not self.samples:
            self.num_features = 0
            return

        all_features = np.array([s["features"] for s in self.samples])
        self.num_features = all_features.shape[1]

        if all_features.size > 0:
            scaled_features = self.feature_scaler.fit_transform(all_features)
            for i, sample in enumerate(self.samples):
                sample["features_scaled"] = torch.tensor(
                    scaled_features[i], dtype=torch.float32
                )
        else:
            for i, sample in enumerate(self.samples):
                sample["features_scaled"] = torch.zeros(
                    self.num_features, dtype=torch.float32
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample["features_scaled"],
            sample["sequence_tokens"],
            torch.tensor(sample["metric"], dtype=torch.float32),
        )


class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PassRanker(nn.Module):
    """
    Transformer-based model to predict performance from pass sequences and features.
    """

    def __init__(
        self,
        vocab_size,
        num_features,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        feature_mlp_layers,
        max_seq_len,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pass_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        mlp_layers = []
        input_dim = num_features
        for layer_dim in feature_mlp_layers:
            mlp_layers.append(nn.Linear(input_dim, layer_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        self.feature_mlp = nn.Sequential(*mlp_layers)

        self.fusion_dim = d_model + (
            feature_mlp_layers[-1] if feature_mlp_layers else 0
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fusion_dim // 2, 1),
        )

    def forward(self, features, sequence_tokens):

        seq_emb = self.pass_embedding(sequence_tokens) * math.sqrt(self.d_model)
        seq_emb = self.pos_encoder(seq_emb)

        padding_mask = sequence_tokens == 0

        transformer_out = self.transformer_encoder(
            seq_emb, src_key_padding_mask=padding_mask
        )

        mask_expanded = ~padding_mask.unsqueeze(-1).expand_as(transformer_out)
        seq_representation = (transformer_out * mask_expanded).sum(
            1
        ) / mask_expanded.sum(1)

        feature_representation = self.feature_mlp(features)

        fused = torch.cat((seq_representation, feature_representation), dim=1)
        prediction = self.regression_head(fused)

        return prediction.squeeze(-1)


import os


def train_model(config):
    """Main function to orchestrate model training."""
    print("--- NeuroOpt PassRanker Training ---")
    print(f"Configuration: {config}")

    log_dir = f'runs/passranker_{config["target_metric"]}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("tools/training_data/training_data_flat.json", "r") as f:
        json_data = json.load(f)

    dataset = PassSequenceDataset(
        data=json_data,
        target_metric=config["target_metric"],
        max_seq_len=config["max_seq_len"],
    )

    if len(dataset) == 0:
        print("Dataset is empty. No data to train on. Exiting.")
        return

    val_size = int(len(dataset) * config["val_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Training on {train_size} samples, validating on {val_size} samples.")

    model = PassRanker(
        vocab_size=dataset.vocab_size,
        num_features=dataset.num_features,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        feature_mlp_layers=config["feature_mlp_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0
        for features, sequences, targets in train_loader:
            features, sequences, targets = (
                features.to(device),
                sequences.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            predictions = model(features, sequences)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, sequences, targets in val_loader:
                features, sequences, targets = (
                    features.to(device),
                    sequences.to(device),
                    targets.to(device),
                )
                predictions = model(features, sequences)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss},{avg_val_loss}\n")

        checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
            },
            checkpoint_path,
        )

        scheduler.step()

    print("--- Training Complete ---")

    model_save_path = f'passranker_{config["target_metric"]}.pth'
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.pass_vocab,
            "feature_keys": dataset.feature_keys,
            "feature_scaler": dataset.feature_scaler,
            "config": config,
        },
        model_save_path,
    )
    print(f"Model and data saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PassRanker model.")
    parser.add_argument(
        "--target_metric",
        type=str,
        default=CONFIG["target_metric"],
        choices=["runtime", "binary_size"],
        help="The target metric to optimize for.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate.")
    parser.add_argument(
        "--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size."
    )

    args = parser.parse_args()

    config = CONFIG.copy()
    config.update(vars(args))

    train_model(config)
