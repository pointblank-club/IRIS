import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import argparse
from pathlib import Path
import math
import os
import joblib  # For saving/loading scaler
from data_preprocessing_hybrid import (
    load_and_preprocess_data,
    MAX_PASS_SEQ_LEN,
    TARGET_METRICS,
)

# --- Configuration ---
# These can be made configurable via argparse if needed
MAX_PASS_SEQ_LEN = 30  # Max length for common_passes and machine_passes
D_MODEL = 128  # Embedding dimension
NHEAD = 4  # Number of attention heads
NUM_ENCODER_LAYERS = 2  # Number of Transformer encoder layers for pass sequences
DIM_FEEDFORWARD = (
    256  # Dimension of the feedforward network model in Transformer encoders
)
FEATURE_MLP_LAYERS = [64, 32]  # Hidden layers for program feature MLP
DROPOUT = 0.1
TARGET_METRICS = ["runtime", "binary_size"]  # What the model will predict

# --- Utility Functions ---


def clean_passes(pass_str):
    """
    Cleans a comma-separated or space-separated pass string into a list of strings.
    Replaces null/empty with ["none"].
    """
    if pass_str is None or (isinstance(pass_str, str) and pass_str.strip() == ""):
        return ["none"]
    if isinstance(pass_str, list):  # Already a list
        return [p.strip() for p in pass_str if p.strip()] or ["none"]
    if isinstance(pass_str, str):
        # Try splitting by comma first, then by space if no commas
        if "," in pass_str:
            passes = [p.strip() for p in pass_str.split(",") if p.strip()]
        else:
            passes = [p.strip() for p in pass_str.split(" ") if p.strip()]
        return passes or ["none"]
    return ["none"]  # Fallback for unexpected types


def build_vocabularies(data_entries):
    """
    Builds vocabularies for common passes, machine passes, and hardware types.
    """
    common_pass_vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    machine_pass_vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    hardware_vocab = {}

    common_pass_idx = len(common_pass_vocab)
    machine_pass_idx = len(machine_pass_vocab)
    hardware_idx = 0

    for entry in data_entries:
        # Common Passes
        common_passes = clean_passes(entry.get("common_passes"))
        for p in common_passes:
            if p not in common_pass_vocab:
                common_pass_vocab[p] = common_pass_idx
                common_pass_idx += 1

        # Machine Passes
        machine_passes = clean_passes(entry.get("machine_passes"))
        for p in machine_passes:
            if p not in machine_pass_vocab:
                machine_pass_vocab[p] = machine_pass_idx
                machine_pass_idx += 1

        # Hardware
        hardware = entry.get("hardware")
        if hardware and hardware not in hardware_vocab:
            hardware_vocab[hardware] = hardware_idx
            hardware_idx += 1

    print(f"Common Pass Vocab Size: {len(common_pass_vocab)}")
    print(f"Machine Pass Vocab Size: {len(machine_pass_vocab)}")
    print(f"Hardware Vocab Size: {len(hardware_vocab)}")

    return common_pass_vocab, machine_pass_vocab, hardware_vocab


def tokenize_and_pad(sequence_list, vocab, max_len):
    """Converts a list of pass names to token IDs, adds SOS/EOS, and pads/truncates."""
    token_ids = [vocab["<sos>"]]
    token_ids.extend([vocab.get(p, vocab["<unk>"]) for p in sequence_list])
    token_ids.append(vocab["<eos>"])

    if len(token_ids) < max_len:
        token_ids.extend([vocab["<pad>"]] * (max_len - len(token_ids)))
    else:
        token_ids = token_ids[:max_len]
        token_ids[-1] = vocab["<eos>"]  # Ensure EOS is at the end if truncated

    return torch.tensor(token_ids, dtype=torch.long)


# --- Dataset Class ---


class HybridPassDataset(Dataset):
    def __init__(self, processed_samples):
        self.samples = processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample["program_features"],
            sample["hardware_id"],
            sample["common_passes_seq"],
            sample["machine_passes_seq"],
            sample["labels"],
        )


# --- Model Architecture ---


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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]  # Add positional encoding
        return self.dropout(x)


class HybridPassFormer(nn.Module):
    """
    Transformer-based regression model for IRis.
    Predicts runtime and binary_size from program features, hardware, and pass sequences.
    """

    def __init__(
        self,
        common_vocab_size,
        machine_vocab_size,
        num_features,
        hardware_vocab_size,
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
        self.max_seq_len = max_seq_len

        # Embeddings
        self.common_pass_embedding = nn.Embedding(common_vocab_size, d_model)
        self.machine_pass_embedding = nn.Embedding(machine_vocab_size, d_model)
        self.hardware_embedding = nn.Embedding(hardware_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Program Feature MLP
        mlp_layers = []
        input_dim = num_features
        for layer_dim in feature_mlp_layers:
            mlp_layers.append(nn.Linear(input_dim, layer_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        self.feature_mlp = nn.Sequential(*mlp_layers)
        self.feature_projection = nn.Linear(
            input_dim, d_model
        )  # Project MLP output to d_model

        # Transformer Encoders for Pass Sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.common_pass_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )
        self.machine_pass_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )  # Shared encoder layers

        # Fusion Head
        # Input to fusion head: common_pass_encoder_output + machine_pass_encoder_output + hardware_embedding + feature_projection
        # We'll take the mean of sequence encoder outputs for simplicity
        fusion_input_dim = (
            d_model * 4
        )  # 2 * encoder_output (mean pooled) + hardware_emb + feature_proj
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regression Head (for runtime and binary_size)
        self.regression_head = nn.Linear(d_model // 2, len(TARGET_METRICS))

    def forward(
        self, program_features, hardware_ids, common_passes_seq, machine_passes_seq
    ):
        # 1. Process Program Features
        feature_representation = self.feature_mlp(program_features)
        feature_proj = self.feature_projection(
            feature_representation
        )  # (batch_size, d_model)

        # 2. Process Hardware ID
        hardware_emb = self.hardware_embedding(hardware_ids)  # (batch_size, d_model)

        # 3. Process Common Pass Sequence
        common_pass_emb = self.common_pass_embedding(
            common_passes_seq
        )  # (batch_size, seq_len, d_model)
        common_pass_emb = self.pos_encoder(common_pass_emb)
        common_pass_output = self.common_pass_encoder(
            common_pass_emb
        )  # (batch_size, seq_len, d_model)
        # Mean pooling over sequence dimension to get a fixed-size representation
        common_pass_pooled = common_pass_output.mean(dim=1)  # (batch_size, d_model)

        # 4. Process Machine Pass Sequence
        machine_pass_emb = self.machine_pass_embedding(
            machine_passes_seq
        )  # (batch_size, seq_len, d_model)
        machine_pass_emb = self.pos_encoder(machine_pass_emb)
        machine_pass_output = self.machine_pass_encoder(
            machine_pass_emb
        )  # (batch_size, seq_len, d_model)
        # Mean pooling
        machine_pass_pooled = machine_pass_output.mean(dim=1)  # (batch_size, d_model)

        # 5. Fuse all representations
        fused_representation = torch.cat(
            [feature_proj, hardware_emb, common_pass_pooled, machine_pass_pooled], dim=1
        )  # (batch_size, d_model * 4)

        fused_representation = self.fusion_head(fused_representation)

        # 6. Regression Head
        predictions = self.regression_head(
            fused_representation
        )  # (batch_size, len(TARGET_METRICS))

        return predictions


# --- Training and Evaluation Functions ---


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for (
        program_features,
        hardware_ids,
        common_passes_seq,
        machine_passes_seq,
        labels,
    ) in dataloader:
        program_features = program_features.to(device)
        hardware_ids = hardware_ids.to(device)
        common_passes_seq = common_passes_seq.to(device)
        machine_passes_seq = machine_passes_seq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(
            program_features, hardware_ids, common_passes_seq, machine_passes_seq
        )
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (
            program_features,
            hardware_ids,
            common_passes_seq,
            machine_passes_seq,
            labels,
        ) in dataloader:
            program_features = program_features.to(device)
            hardware_ids = hardware_ids.to(device)
            common_passes_seq = common_passes_seq.to(device)
            machine_passes_seq = machine_passes_seq.to(device)
            labels = labels.to(device)

            predictions = model(
                program_features, hardware_ids, common_passes_seq, machine_passes_seq
            )
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid PassFormer model for IRis."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the input flattened JSON dataset (e.g., new_flattened_hybrid_data.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save model, vocabularies, and scaler.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load a few samples and print shapes/tokens without training.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preprocessing and Dataset Creation ---
    print("Loading data and building vocabularies...")
    with open(args.input_json, "r") as f:
        raw_data_entries = json.load(f)

    # Build vocabularies from raw data
    common_pass_vocab, machine_pass_vocab, hardware_vocab = build_vocabularies(
        raw_data_entries
    )

    # Create dataset (feature_scaler and feature_keys will be fitted/discovered here)
    dataset = HybridPassDataset(
        args.input_json,
        common_pass_vocab,
        machine_pass_vocab,
        hardware_vocab,
        MAX_PASS_SEQ_LEN,
        target_metrics=TARGET_METRICS,
    )

    # Save vocabularies, scaler, and feature keys
    with open(output_path / "common_pass_vocab.json", "w") as f:
        json.dump(common_pass_vocab, f, indent=2)
    with open(output_path / "machine_pass_vocab.json", "w") as f:
        json.dump(machine_pass_vocab, f, indent=2)
    with open(output_path / "hardware_vocab.json", "w") as f:
        json.dump(hardware_vocab, f, indent=2)
    joblib.dump(dataset.feature_scaler, output_path / "feature_scaler.pkl")
    with open(output_path / "feature_keys.json", "w") as f:
        json.dump(dataset.feature_keys, f, indent=2)

    print("Vocabularies, scaler, and feature keys saved.")

    if args.dry_run:
        print("\n--- Dry Run Mode ---")
        sample_dataloader = DataLoader(
            dataset, batch_size=min(4, len(dataset)), shuffle=False
        )
        for i, (
            program_features,
            hardware_ids,
            common_passes_seq,
            machine_passes_seq,
            labels,
        ) in enumerate(sample_dataloader):
            print(f"\nSample Batch {i+1}:")
            print(f"  Program Features shape: {program_features.shape}")
            print(f"  Hardware IDs shape: {hardware_ids.shape}")
            print(f"  Common Passes Seq shape: {common_passes_seq.shape}")
            print(f"  Machine Passes Seq shape: {machine_passes_seq.shape}")
            print(f"  Labels shape: {labels.shape}")

            # Decode a few sequences for verification
            id_to_common_pass = {v: k for k, v in common_pass_vocab.items()}
            id_to_machine_pass = {v: k for k, v in machine_pass_vocab.items()}
            id_to_hardware = {v: k for k, v in hardware_vocab.items()}

            print(
                f"  Decoded Common Passes (first sample): {[id_to_common_pass.get(idx.item(), '<unk>') for idx in common_passes_seq[0] if idx.item() not in [0,2,3]]}"
            )
            print(
                f"  Decoded Machine Passes (first sample): {[id_to_machine_pass.get(idx.item(), '<unk>') for idx in machine_passes_seq[0] if idx.item() not in [0,2,3]]}"
            )
            print(
                f"  Decoded Hardware (first sample): {id_to_hardware.get(hardware_ids[0].item(), '<unk>')}"
            )
            print(f"  Labels (first sample): {labels[0].tolist()}")
            if i >= 0:  # Just print one batch for dry run
                break
        print("\nDry run complete. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- Model Initialization ---
    print("\nInitializing model...")
    model = HybridPassFormer(
        common_vocab_size=len(common_pass_vocab),
        machine_vocab_size=len(machine_pass_vocab),
        num_features=len(dataset.feature_keys),
        hardware_vocab_size=len(hardware_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        feature_mlp_layers=FEATURE_MLP_LAYERS,
        max_seq_len=MAX_PASS_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # For regression on runtime and binary_size

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        # In a real scenario, you'd split into train/val and evaluate
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")

    print("\nTraining complete. Saving model...")

    # --- Save Model and Components ---
    model_save_path = output_path / "hybrid_passformer_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "common_pass_vocab": common_pass_vocab,
            "machine_pass_vocab": machine_pass_vocab,
            "hardware_vocab": hardware_vocab,
            "feature_scaler": dataset.feature_scaler,
            "feature_keys": dataset.feature_keys,
            "config": {
                "common_vocab_size": len(common_pass_vocab),
                "machine_vocab_size": len(machine_pass_vocab),
                "num_features": len(dataset.feature_keys),
                "hardware_vocab_size": len(hardware_vocab),
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_encoder_layers": NUM_ENCODER_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "feature_mlp_layers": FEATURE_MLP_LAYERS,
                "max_seq_len": MAX_PASS_SEQ_LEN,
                "dropout": DROPOUT,
                "target_metrics": TARGET_METRICS,
            },
        },
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")


# --- TODOs for Future Improvement ---
# TODO: Implement masking for invalid passes for a specific hardware type.
#       This would involve creating a mask based on hardware_id and applying it
#       within the common_pass_encoder and machine_pass_encoder or during fusion.
#       For example, certain passes might not be valid for 'tpu' hardware.
# TODO: Explore different fusion strategies for encoder outputs (e.g., learned gating, attention).
# TODO: Implement a proper train/validation split for robust evaluation.
# TODO: Add early stopping and learning rate scheduling.
# TODO: Consider using a separate regression head for each target metric or a multi-task loss.
# TODO: Refine the 'clean_passes' logic if pass sequences can contain complex structures or special characters.
# TODO: Add a mechanism to handle unknown hardware types during inference.

if __name__ == "__main__":
    main()
