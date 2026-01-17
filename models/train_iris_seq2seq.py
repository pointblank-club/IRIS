import json
import torch
from torch.utils.data import DataLoader, random_split
import argparse
from neuropt import PassSequenceDataset, PassFormer, beam_search_decode
from nltk.translate.bleu_score import sentence_bleu
from Levenshtein import distance as levenshtein_distance
from scipy.spatial.distance import jaccard


CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_decoder_layers": 2,
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


def evaluate_model(model, val_dataset, device):
    """Evaluates the model on the validation set using multiple metrics."""
    model.eval()
    total_bleu_score = 0
    exact_matches = 0
    total_jaccard_similarity = 0
    total_levenshtein_distance = 0
    dataset = val_dataset.dataset

    print("\n--- Model Evaluation (Beam Search) ---")
    for idx in val_dataset.indices:
        original_sample = dataset.samples[idx]
        features = original_sample["features"]

        generated_sequence = beam_search_decode(
            model, features, dataset.feature_scaler, dataset.pass_vocab, device
        )
        reference_sequence = original_sample["sequence"]

        print(f"Reference sequence: {reference_sequence}")
        print(f"Generated sequence: {generated_sequence}")

        if generated_sequence == reference_sequence:
            exact_matches += 1

        jaccard_sim = 1 - jaccard(set(reference_sequence), set(generated_sequence))
        total_jaccard_similarity += jaccard_sim

        lev_dist = levenshtein_distance(generated_sequence, reference_sequence)
        total_levenshtein_distance += lev_dist

        bleu_score = sentence_bleu([reference_sequence], generated_sequence)
        total_bleu_score += bleu_score

        print(
            f"BLEU: {bleu_score:.4f}, Jaccard: {jaccard_sim:.4f}, Levenshtein: {lev_dist}"
        )

    num_samples = len(val_dataset)
    avg_bleu_score = total_bleu_score / num_samples
    exact_match_accuracy = exact_matches / num_samples
    avg_jaccard_similarity = total_jaccard_similarity / num_samples
    avg_levenshtein_distance = total_levenshtein_distance / num_samples

    print("\n--- Evaluation Summary ---")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Average Jaccard Similarity: {avg_jaccard_similarity:.4f}")
    print(f"Average Levenshtein Distance: {avg_levenshtein_distance:.4f}")
    print(f"Average BLEU score: {avg_bleu_score:.4f}")


import os


def train_model(config):
    """Main function to orchestrate model training."""
    print("--- NeuroOpt PassFormer Training ---")
    print(f"Configuration: {config}")

    log_dir = f'runs/passformer_{config["target_metric"]}'
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

    model = PassFormer(
        vocab_size=dataset.vocab_size,
        num_features=dataset.num_features,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        feature_mlp_layers=config["feature_mlp_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.pass_vocab["<pad>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0
        for features, sequences in train_loader:
            features, sequences = features.to(device), sequences.to(device)

            decoder_input = sequences[:, :-1]
            decoder_target = sequences[:, 1:]

            optimizer.zero_grad()
            predictions = model(features, decoder_input)

            loss = loss_fn(
                predictions.reshape(-1, dataset.vocab_size), decoder_target.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, sequences in val_loader:
                features, sequences = features.to(device), sequences.to(device)

                decoder_input = sequences[:, :-1]
                decoder_target = sequences[:, 1:]

                predictions = model(features, decoder_input)
                loss = loss_fn(
                    predictions.reshape(-1, dataset.vocab_size),
                    decoder_target.reshape(-1),
                )
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

    model_save_path = f'passformer_{config["target_metric"]}.pth'
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

    evaluate_model(model, val_dataset, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PassFormer model.")
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
