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
import random
from collections import Counter
import joblib

# --- Configuration ---
MAX_PASS_SEQ_LEN = 60
TARGET_METRICS = ["execution_time"]
TARGET_METRICS_ALL = ["execution_time", "binary_size"]

D_MODEL = 128
NHEAD = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 256
FEATURE_MLP_LAYERS = [64, 32]
DROPOUT = 0.1
CONTEXT_TOKENS = 7

# --- Utility Functions ---


def calculate_ngram_overlap(
    predicted_sequences, target_sequences, n_grams=[1, 2, 3, 4]
):
    """Calculate n-gram overlap between predicted and target sequences."""
    overlap_scores = {f"{n}-gram_overlap": 0.0 for n in n_grams}
    total_samples = 0

    for pred_seq, target_seq in zip(predicted_sequences, target_sequences):
        if not target_seq:
            continue
        total_samples += 1

        for n in n_grams:
            if len(pred_seq) < n or len(target_seq) < n:
                continue

            pred_ngrams = set(
                tuple(pred_seq[i : i + n]) for i in range(len(pred_seq) - n + 1)
            )
            target_ngrams = set(
                tuple(target_seq[i : i + n]) for i in range(len(target_seq) - n + 1)
            )

            if len(target_ngrams) == 0:
                overlap_scores[f"{n}-gram_overlap"] += 0.0
            else:
                overlap = len(pred_ngrams.intersection(target_ngrams))
                overlap_scores[f"{n}-gram_overlap"] += overlap / len(target_ngrams)

    for n_gram_key in overlap_scores:
        overlap_scores[n_gram_key] /= total_samples if total_samples > 0 else 1.0

    return overlap_scores


def calculate_sequence_accuracy(predicted_sequences, target_sequences):
    """Calculate exact sequence match accuracy."""
    correct = 0
    total = len(predicted_sequences)

    for pred_seq, target_seq in zip(predicted_sequences, target_sequences):
        if pred_seq == target_seq:
            correct += 1

    return correct / total if total > 0 else 0.0


# --- Hardware-aware masking ---


def build_allowed_token_mask(hardware_ids, joint_pass_vocab, hardware_vocab, device):
    """
    Returns a boolean mask of shape [B, V] indicating which tokens are allowed for each sample
    given the hardware id. Allowed:
      - All common (un-tagged) pass tokens
      - Machine-tagged tokens whose prefix matches the hardware string
    """
    # Invert vocabularies
    id_to_hardware = {v: k for k, v in hardware_vocab.items()}
    id_to_pass = {v: k for k, v in joint_pass_vocab.items()}

    vocab_size = len(joint_pass_vocab)

    # Precompute common vs tagged tokens once
    common_token_indices = []
    tagged_prefix_by_index = {}
    for tok_id in range(vocab_size):
        name = id_to_pass.get(tok_id, "")
        if name in ("<pad>", "<unk>", "<sos>", "<eos>"):
            common_token_indices.append(tok_id)
        elif "::" in name:
            prefix = name.split("::", 1)[0]
            tagged_prefix_by_index[tok_id] = prefix
        else:
            common_token_indices.append(tok_id)

    batch_size = hardware_ids.size(0)
    mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    if batch_size == 0:
        return mask

    # Unique hardware ids to build per-hardware masks
    unique_hids = torch.unique(hardware_ids).tolist()
    per_hw_mask = {}
    for hid in unique_hids:
        hw_str = id_to_hardware.get(int(hid), "").lower()
        hw_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        # Allow common tokens
        if common_token_indices:
            hw_mask[torch.tensor(common_token_indices, device=device)] = True
        # Allow tagged tokens for matching prefix
        if hw_str:
            for tok_id, prefix in tagged_prefix_by_index.items():
                if prefix == hw_str:
                    hw_mask[tok_id] = True
        per_hw_mask[int(hid)] = hw_mask

    # Assemble batch mask
    for i in range(batch_size):
        hid = int(hardware_ids[i].item())
        mask[i] = per_hw_mask.get(
            hid, torch.zeros(vocab_size, dtype=torch.bool, device=device)
        )

    return mask


# --- Dataset Class ---


class PassGenDataset(Dataset):
    def __init__(
        self,
        processed_samples,
        feature_scaler=None,
        target_metric_scaler=None,
        pad_id=None,
    ):
        self.samples = processed_samples
        self.feature_scaler = feature_scaler
        self.target_metric_scaler = target_metric_scaler
        self.pad_id = pad_id

        # Pre-calculate index of the target metric in the labels array
        try:
            self.target_metric_idx = TARGET_METRICS_ALL.index(TARGET_METRICS[0])
        except ValueError:
            raise ValueError(
                f"Target metric {TARGET_METRICS[0]} not found in {TARGET_METRICS_ALL}"
            )

    def __len__(self):
        return len(self.samples)

    def set_scalers(self, feature_scaler, target_metric_scaler):
        self.feature_scaler = feature_scaler
        self.target_metric_scaler = target_metric_scaler

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.array(value, dtype=np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features = self._to_numpy(sample["program_features"])
        if self.feature_scaler is not None and features.size:
            features = self.feature_scaler.transform(features.reshape(1, -1))[0]
        program_features = torch.tensor(features, dtype=torch.float32)

        # Extract only the relevant target metric label
        labels = self._to_numpy(sample["labels"])
        labels = labels[
            self.target_metric_idx : self.target_metric_idx + 1
        ]  # Keep as 1-element array

        if self.target_metric_scaler is not None and labels.size:
            labels = self.target_metric_scaler.transform(labels.reshape(1, -1))[0]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return (
            program_features,
            sample["hardware_id"],
            sample["input_sequence"],
            sample["target_sequence"],
            labels_tensor,
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PassGenTransformer(nn.Module):
    """Transformer-based sequence generation model for compiler passes."""

    def __init__(
        self,
        vocab_size,
        num_features,
        hardware_vocab_size,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        feature_mlp_layers,
        max_seq_len,
        dropout=0.1,
        context_tokens=CONTEXT_TOKENS,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.context_tokens = context_tokens

        # Embeddings
        self.pass_embedding = nn.Embedding(vocab_size, d_model)
        self.hardware_embedding = nn.Embedding(hardware_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        self.sequence_dropout = nn.Dropout(dropout)

        # Program Feature MLP
        mlp_layers = []
        input_dim = num_features
        for layer_dim in feature_mlp_layers:
            mlp_layers.append(nn.Linear(input_dim, layer_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        self.feature_mlp = nn.Sequential(*mlp_layers)
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.context_projection = nn.Linear(input_dim, d_model * self.context_tokens)
        self.context_pos_encoder = PositionalEncoding(
            d_model, dropout, self.context_tokens + 2
        )
        self.context_dropout = nn.Dropout(dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )

        # Output layer
        self.output_head = nn.Linear(d_model, vocab_size)

        # Auxiliary Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),  # Output dimension is 1 for a single metric
        )

    def _build_context(self, program_features, hardware_ids):
        """Encode program features and hardware into decoder memory tokens."""
        batch_size = program_features.size(0)
        feature_representation = self.feature_mlp(program_features)
        feature_proj = self.feature_projection(feature_representation)
        feature_tokens = self.context_projection(feature_representation)
        feature_tokens = feature_tokens.view(
            batch_size, self.context_tokens, self.d_model
        )

        hardware_emb = self.hardware_embedding(hardware_ids)

        context_tokens = torch.cat(
            [feature_proj.unsqueeze(1), feature_tokens, hardware_emb.unsqueeze(1)],
            dim=1,
        )
        context_tokens = self.context_pos_encoder(context_tokens)
        context_tokens = self.context_dropout(context_tokens)

        regression_input = feature_proj + hardware_emb
        return context_tokens, regression_input

    def forward(
        self,
        program_features,
        hardware_ids,
        input_sequence,
        tgt_key_padding_mask=None,
        allowed_token_mask=None,
    ):
        context, regression_input = self._build_context(program_features, hardware_ids)

        # Process Input Sequence
        input_emb = self.pass_embedding(input_sequence)
        input_emb = self.pos_encoder(input_emb)
        input_emb = self.sequence_dropout(input_emb)

        # Generate causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            input_sequence.size(1)
        ).to(input_sequence.device)

        # Transformer Decoder
        decoder_output = self.transformer_decoder(
            input_emb,
            context,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Output predictions
        output = self.output_head(decoder_output)
        # Apply hardware-aware mask to logits if provided: disallow tokens by setting -inf
        if allowed_token_mask is not None:
            # allowed_token_mask: [B, V] -> expand to [B, T, V]
            B, T, V = output.shape
            expanded = allowed_token_mask.unsqueeze(1).expand(B, T, V)
            output = output.masked_fill(~expanded, -1e9)

        # Auxiliary metrics prediction
        predicted_metrics = self.regression_head(regression_input)

        return output, predicted_metrics


def _generate_sequence_greedy(
    self,
    program_features,
    hardware_ids,
    start_token,
    end_token,
    pad_token,
    device,
    max_len=MAX_PASS_SEQ_LEN,
    allowed_token_mask=None,
):
    """Autoregressive greedy decoding for inference."""
    self.eval()
    batch_size = program_features.size(0)

    context, _ = self._build_context(program_features, hardware_ids)
    generated_sequences = torch.full(
        (batch_size, 1), start_token, dtype=torch.long, device=device
    )

    with torch.no_grad():
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            input_emb = self.pass_embedding(generated_sequences)
            input_emb = self.pos_encoder(input_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated_sequences.size(1)
            ).to(device)
            decoder_output = self.transformer_decoder(
                input_emb, context, tgt_mask=tgt_mask
            )
            next_token_logits = self.output_head(decoder_output[:, -1, :])
            if allowed_token_mask is not None:
                next_token_logits = next_token_logits.masked_fill(
                    ~allowed_token_mask, -1e9
                )
            next_token_logits[:, pad_token] = -1e9
            next_token_logits[:, start_token] = -1e9

            # Repetition penalties to discourage degenerative loops
            if generated_sequences.size(1) > 1:
                for b in range(batch_size):
                    seq_tokens = generated_sequences[b].tolist()
                    token_counts = Counter(seq_tokens)
                    for tok, count in token_counts.items():
                        if tok in (pad_token, start_token, end_token):
                            continue
                        if count >= 3:
                            next_token_logits[b, tok] -= 0.5 * (count - 2)
                    last_tok = seq_tokens[-1]
                    if last_tok not in (pad_token, start_token, end_token):
                        next_token_logits[b, last_tok] -= 0.3

            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            next_tokens[finished] = pad_token
            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)

            finished = finished | (next_tokens.squeeze(-1) == end_token)
            if finished.all():
                break

    if generated_sequences.size(1) < max_len:
        padding = torch.full(
            (batch_size, max_len - generated_sequences.size(1)),
            pad_token,
            dtype=torch.long,
            device=device,
        )
        generated_sequences = torch.cat([generated_sequences, padding], dim=1)

    return generated_sequences[:, :max_len]


# --- Training and Evaluation Functions ---


def _generate_sequence_beam(
    self,
    program_features,
    hardware_ids,
    start_token,
    end_token,
    pad_token,
    device,
    max_len=MAX_PASS_SEQ_LEN,
    beam_size=5,
    length_penalty=0.6,
    min_len: int = 3,
    repetition_penalty: float = 1.2,
    allowed_token_mask=None,
):
    """Beam search decoding. Returns tensor [B, max_len]."""
    self.eval()
    batch_size = program_features.size(0)

    with torch.no_grad():
        context, _ = self._build_context(program_features, hardware_ids)

        results = []
        for b in range(batch_size):
            ctx = context[b : b + 1]
            allowed_mask_b = None
            if allowed_token_mask is not None:
                allowed_mask_b = allowed_token_mask[b : b + 1].clone()
            beams = [
                (
                    torch.tensor([[start_token]], device=device, dtype=torch.long),
                    0.0,
                    False,
                )
            ]

            for _ in range(max_len - 1):
                new_beams = []
                all_finished = True
                for seq, score, finished in beams:
                    if finished:
                        new_beams.append((seq, score, True))
                        continue
                    all_finished = False
                    input_emb = self.pass_embedding(seq)
                    input_emb = self.pos_encoder(input_emb)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        seq.size(1)
                    ).to(device)
                    dec_out = self.transformer_decoder(
                        input_emb, ctx, tgt_mask=tgt_mask
                    )
                    logits = self.output_head(dec_out[:, -1, :])
                    logits[:, pad_token] = -1e9
                    logits[:, start_token] = -1e9
                    if seq.size(1) < min_len:
                        logits[:, end_token] = -1e9
                    if allowed_mask_b is not None:
                        logits = logits.masked_fill(~allowed_mask_b, -1e9)
                    if repetition_penalty is not None and repetition_penalty > 1.0:
                        seen = torch.unique(seq)
                        logits[:, seen] -= math.log(repetition_penalty)
                    if seq.size(1) > 1:
                        seq_tokens = seq.view(-1).tolist()
                        token_counts = Counter(seq_tokens)
                        for tok, count in token_counts.items():
                            if tok in (pad_token, start_token, end_token):
                                continue
                            if count >= 3:
                                logits[0, tok] -= 0.5 * (count - 2)
                        last_tok = seq_tokens[-1]
                        if last_tok not in (pad_token, start_token, end_token):
                            logits[0, last_tok] -= 0.3
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_logp, topk_idx = torch.topk(log_probs, k=beam_size, dim=-1)
                    for k in range(beam_size):
                        nt = topk_idx[0, k].view(1, 1)
                        nlp = topk_logp[0, k].item()
                        new_seq = torch.cat([seq, nt], dim=1)
                        length_norm = ((5 + new_seq.size(1)) / 6) ** length_penalty
                        new_score = (score + nlp) / length_norm
                        finished_flag = nt.item() == end_token
                        new_beams.append((new_seq, new_score, finished_flag))
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]
                if all_finished:
                    break

            best_seq = max(beams, key=lambda x: x[1])[0]
            if best_seq.size(1) < max_len:
                pad_tail = torch.full(
                    (1, max_len - best_seq.size(1)),
                    pad_token,
                    dtype=torch.long,
                    device=device,
                )
                best_seq = torch.cat([best_seq, pad_tail], dim=1)
            results.append(best_seq[:, :max_len])

        return torch.cat(results, dim=0)


# Attach helper as class method to avoid indentation collisions.
PassGenTransformer.generate_sequence_greedy = _generate_sequence_greedy
PassGenTransformer.generate_sequence_beam = _generate_sequence_beam


def sample_random_prefix_batch(
    input_sequence, target_sequence, pad_id, min_prefix_tokens=1, max_prefix_tokens=None
):
    """Randomly truncate sequences to expose many prefix→next-token examples.

    With almost-unique 60-token targets, full-length teacher forcing provides
    sparse supervision. Sampling prefixes each iteration teaches the decoder
    local transitions without discarding samples. A curriculum can limit the
    maximum prefix length via ``max_prefix_tokens``.
    """
    if min_prefix_tokens < 1:
        min_prefix_tokens = 1

    batch_size = input_sequence.size(0)
    prefix_inputs = torch.full_like(input_sequence, pad_id)
    prefix_targets = torch.full_like(target_sequence, pad_id)

    input_cpu = input_sequence.detach().cpu()
    target_cpu = target_sequence.detach().cpu()

    for idx in range(batch_size):
        valid_len = int((target_cpu[idx] != pad_id).sum().item())
        if valid_len <= 0:
            prefix_len = 1
        else:
            upper = (
                valid_len
                if max_prefix_tokens is None
                else min(valid_len, max_prefix_tokens)
            )
            upper = max(upper, min_prefix_tokens)
            prefix_len = random.randint(min_prefix_tokens, upper)

        prefix_inputs[idx, :prefix_len] = input_cpu[idx, :prefix_len]
        prefix_targets[idx, :prefix_len] = target_cpu[idx, :prefix_len]

    return prefix_inputs.to(input_sequence.device), prefix_targets.to(
        target_sequence.device
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    regression_criterion,
    regression_loss_weight,
    device,
    pad_id,
    joint_pass_vocab,
    hardware_vocab,
    next_token_mode=False,
    min_prefix_tokens=1,
    max_prefix_tokens=None,
):
    model.train()
    total_loss = 0
    total_seq_loss = 0
    total_reg_loss = 0

    for (
        program_features,
        hardware_ids,
        input_sequence,
        target_sequence,
        labels,
    ) in dataloader:
        program_features = program_features.to(device)
        hardware_ids = hardware_ids.to(device)
        input_sequence = input_sequence.to(device)
        target_sequence = target_sequence.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        seq_inputs = input_sequence
        seq_targets = target_sequence
        if next_token_mode:
            seq_inputs, seq_targets = sample_random_prefix_batch(
                input_sequence,
                target_sequence,
                pad_id,
                min_prefix_tokens=min_prefix_tokens,
                max_prefix_tokens=max_prefix_tokens,
            )

        tgt_key_padding_mask = seq_inputs == pad_id
        allowed_mask = build_allowed_token_mask(
            hardware_ids, joint_pass_vocab, hardware_vocab, device
        )
        predictions, predicted_metrics = model(
            program_features,
            hardware_ids,
            seq_inputs,
            tgt_key_padding_mask=tgt_key_padding_mask,
            allowed_token_mask=allowed_mask,
        )

        seq_targets = seq_targets.long()
        assert (
            predictions.view(-1, model.vocab_size).shape[0]
            == seq_targets.view(-1).shape[0]
        ), f"Shape mismatch: predictions {predictions.view(-1, model.vocab_size).shape} vs target {seq_targets.view(-1).shape}"

        # Sequence generation loss
        seq_gen_loss = criterion(
            predictions.view(-1, model.vocab_size), seq_targets.view(-1)
        )

        # Regression loss
        reg_loss = regression_criterion(predicted_metrics, labels)

        # Combined loss
        loss = seq_gen_loss + (regression_loss_weight * reg_loss)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seq_loss += seq_gen_loss.item()
        total_reg_loss += reg_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_seq_loss = total_seq_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)

    return avg_loss, avg_seq_loss, avg_reg_loss


def evaluate_epoch(
    model,
    dataloader,
    criterion,
    regression_criterion,
    regression_loss_weight,
    device,
    joint_pass_vocab,
    target_metric_scaler,
    hardware_vocab,
):
    model.eval()
    total_loss = 0
    all_predicted_sequences = []
    all_target_sequences = []
    all_predicted_metrics = []
    all_target_labels = []
    # Token-level accuracy accumulators
    pad_id = joint_pass_vocab["<pad>"]
    token_correct = 0
    token_total = 0

    with torch.no_grad():
        for (
            program_features,
            hardware_ids,
            input_sequence,
            target_sequence,
            labels,
        ) in dataloader:
            program_features = program_features.to(device)
            hardware_ids = hardware_ids.to(device)
            input_sequence = input_sequence.to(device)
            target_sequence = target_sequence.to(device)
            labels = labels.to(device)

            # Forward pass
            tgt_key_padding_mask = input_sequence == joint_pass_vocab["<pad>"]
            allowed_mask = build_allowed_token_mask(
                hardware_ids, joint_pass_vocab, hardware_vocab, device
            )
            predictions, predicted_metrics = model(
                program_features,
                hardware_ids,
                input_sequence,
                tgt_key_padding_mask=tgt_key_padding_mask,
                allowed_token_mask=allowed_mask,
            )
            seq_gen_loss = criterion(
                predictions.view(-1, model.vocab_size), target_sequence.view(-1)
            )
            reg_loss = regression_criterion(predicted_metrics, labels)
            loss = seq_gen_loss + (regression_loss_weight * reg_loss)
            total_loss += loss.item()

            # Teacher-forced token accuracy (ignore pads)
            predicted_tokens = predictions.argmax(dim=-1)
            mask = target_sequence != pad_id
            token_correct += (predicted_tokens.eq(target_sequence) & mask).sum().item()
            token_total += mask.sum().item()

            # Generate sequences for evaluation
            generated_sequences = model.generate_sequence_greedy(
                program_features,
                hardware_ids,
                joint_pass_vocab["<sos>"],
                joint_pass_vocab["<eos>"],
                joint_pass_vocab["<pad>"],
                device,
                MAX_PASS_SEQ_LEN,
                allowed_token_mask=allowed_mask,
            )

            all_predicted_sequences.extend(generated_sequences.tolist())
            all_target_sequences.extend(target_sequence.tolist())
            all_predicted_metrics.extend(predicted_metrics.tolist())
            all_target_labels.extend(labels.tolist())

    # Filter sequences for evaluation
    id_to_pass = {v: k for k, v in joint_pass_vocab.items()}
    filtered_predicted = []
    filtered_target = []

    for pred_seq, target_seq in zip(all_predicted_sequences, all_target_sequences):
        # Remove special tokens
        special_tokens = [
            joint_pass_vocab["<pad>"],
            joint_pass_vocab["<sos>"],
            joint_pass_vocab["<eos>"],
        ]
        filtered_pred = [
            id_to_pass.get(token, "<unk>")
            for token in pred_seq
            if token not in special_tokens
        ]
        filtered_tgt = [
            id_to_pass.get(token, "<unk>")
            for token in target_seq
            if token not in special_tokens
        ]

        filtered_predicted.append(filtered_pred)
        filtered_target.append(filtered_tgt)

    # Calculate metrics
    ngram_overlap_scores = calculate_ngram_overlap(
        filtered_predicted, filtered_target, n_grams=[1, 2, 3, 4]
    )
    # Token-level accuracy from teacher-forced logits
    seq_token_accuracy = (token_correct / token_total) if token_total > 0 else 0.0

    # Unscale regression metrics
    all_target_labels_unscaled = target_metric_scaler.inverse_transform(
        np.array(all_target_labels)
    )
    all_predicted_metrics_unscaled = target_metric_scaler.inverse_transform(
        np.array(all_predicted_metrics)
    )

    # Calculate MAE
    runtime_mae = np.mean(
        np.abs(all_predicted_metrics_unscaled[:, 0] - all_target_labels_unscaled[:, 0])
    )

    return {
        "loss": total_loss / len(dataloader),
        "ngram_overlap": ngram_overlap_scores,
        "seq_accuracy": seq_token_accuracy,
        "runtime_mae": runtime_mae,
    }


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(description="Train PassGen Transformer model.")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the input flattened JSON dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models_seqgen",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--regression_loss_weight",
        type=float,
        default=0.001,
        help="Weight for regression loss.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for sequence loss (0.0 disables).",
    )
    parser.add_argument(
        "--d_model", type=int, default=128, help="Dimension of the model (d_model)."
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads in the Transformer.",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=4,
        help="Number of decoder layers in the Transformer.",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=256,
        help="Dimension of the feedforward network in the Transformer.",
    )
    parser.add_argument(
        "--feature_mlp_layers",
        type=int,
        nargs="+",
        default=[64, 32],
        help="List of layer dimensions for the feature MLP.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument(
        "--next_token_mode",
        action="store_true",
        help="Enable random prefix next-token training to emphasise subsequence patterns.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of initial epochs to suppress regression loss for decoder warm-up.",
    )
    parser.add_argument(
        "--curriculum_epochs",
        type=int,
        default=10,
        help="Epochs over which to ramp prefix length from curriculum_min_len to full length.",
    )
    parser.add_argument(
        "--curriculum_min_len",
        type=int,
        default=8,
        help="Minimum prefix length used at the start of the curriculum.",
    )
    parser.add_argument(
        "--min_prefix_len",
        type=int,
        default=4,
        help="Lower bound for sampled prefix lengths during next-token training.",
    )

    args = parser.parse_args()

    # Hardcode TARGET_METRICS for this specific training run
    global TARGET_METRICS
    TARGET_METRICS = ["execution_time"]

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load preprocessing artifacts and process samples
    print("Loading and preprocessing data...")
    try:
        from data_preprocessing_hybrid import load_and_preprocess_data

        # Pass TARGET_METRICS_ALL to load_and_preprocess_data to ensure all labels are loaded
        processed_samples, _, feature_keys, joint_pass_vocab, hardware_vocab, _ = (
            load_and_preprocess_data(
                args.input_json,
                MAX_PASS_SEQ_LEN,
                ["execution_time", "binary_size"],
                scale=False,
            )
        )

        # Save preprocessing artifacts
        preprocessing_output_path = Path("preprocessing_output")
        preprocessing_output_path.mkdir(parents=True, exist_ok=True)

        with open(preprocessing_output_path / "joint_pass_vocab.json", "w") as f:
            json.dump(joint_pass_vocab, f, indent=2)
        with open(preprocessing_output_path / "hardware_vocab.json", "w") as f:
            json.dump(hardware_vocab, f, indent=2)
        with open(preprocessing_output_path / "feature_keys.json", "w") as f:
            json.dump(feature_keys, f, indent=2)

        # Scalers will be fitted on the training split to avoid leakage and saved after fitting

        print(f"Preprocessing artifacts saved to {preprocessing_output_path}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create dataset
    print("Creating dataset...")
    full_dataset = PassGenDataset(processed_samples, pad_id=joint_pass_vocab["<pad>"])

    # Train/validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Fit scalers using only the training subset
    train_indices = train_dataset.indices
    train_features = (
        np.stack([processed_samples[i]["program_features"] for i in train_indices])
        if train_indices
        else np.empty((0, len(feature_keys)), dtype=np.float32)
    )

    # Extract only the selected target metric for scaling
    target_metric_idx = TARGET_METRICS_ALL.index(TARGET_METRICS[0])
    train_labels = (
        np.stack(
            [
                processed_samples[i]["labels"][
                    target_metric_idx : target_metric_idx + 1
                ]
                for i in train_indices
            ]
        )
        if train_indices
        else np.empty((0, 1), dtype=np.float32)
    )

    feature_scaler = (
        StandardScaler().fit(train_features)
        if train_features.size
        else StandardScaler()
    )
    target_metric_scaler = (
        StandardScaler().fit(train_labels) if train_labels.size else StandardScaler()
    )

    full_dataset.set_scalers(feature_scaler, target_metric_scaler)

    joblib.dump(feature_scaler, preprocessing_output_path / "feature_scaler.pkl")
    joblib.dump(
        target_metric_scaler,
        preprocessing_output_path / f"target_metric_scaler_{TARGET_METRICS[0]}.pkl",
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Initialize model
    print("Initializing model...")
    model = PassGenTransformer(
        vocab_size=len(joint_pass_vocab),
        num_features=len(feature_keys),
        hardware_vocab_size=len(hardware_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        feature_mlp_layers=args.feature_mlp_layers,
        max_seq_len=MAX_PASS_SEQ_LEN,
        dropout=args.dropout,
        context_tokens=CONTEXT_TOKENS,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=joint_pass_vocab["<pad>"],
        label_smoothing=max(0.0, min(0.99, args.label_smoothing)),
    )
    regression_criterion = nn.MSELoss()

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    pad_id = joint_pass_vocab["<pad>"]

    for epoch in range(args.epochs):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            effective_reg_weight = 0.0
        else:
            effective_reg_weight = args.regression_loss_weight

        if args.curriculum_epochs > 0 and epoch < args.curriculum_epochs:
            ratio = (epoch + 1) / args.curriculum_epochs
            target_max = MAX_PASS_SEQ_LEN
            curriculum_min = min(args.curriculum_min_len, target_max)
            max_prefix = int(curriculum_min + ratio * (target_max - curriculum_min))
            max_prefix = max(max_prefix, curriculum_min)
        else:
            max_prefix = None

        train_loss, train_seq_loss, train_reg_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            regression_criterion,
            effective_reg_weight,
            device,
            pad_id,
            joint_pass_vocab,
            hardware_vocab,
            next_token_mode=args.next_token_mode,
            min_prefix_tokens=max(args.min_prefix_len, 1),
            max_prefix_tokens=max_prefix,
        )

        val_metrics = evaluate_epoch(
            model,
            val_dataloader,
            criterion,
            regression_criterion,
            effective_reg_weight,
            device,
            joint_pass_vocab,
            target_metric_scaler,
            hardware_vocab,
        )

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(
            f"  Train Loss: {train_loss:.4f} (Seq: {train_seq_loss:.4f}, Reg: {train_reg_loss:.4f})"
        )
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Seq Token Acc: {val_metrics['seq_accuracy']:.4f}")
        print(f"  1-gram overlap: {val_metrics['ngram_overlap']['1-gram_overlap']:.4f}")
        print(f"  2-gram overlap: {val_metrics['ngram_overlap']['2-gram_overlap']:.4f}")
        print(f"  Runtime MAE: {val_metrics['runtime_mae']:.4f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            print("  -> Saving best model")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "joint_pass_vocab": joint_pass_vocab,
                    "hardware_vocab": hardware_vocab,
                    "feature_keys": feature_keys,
                    "config": {
                        "vocab_size": len(joint_pass_vocab),
                        "num_features": len(feature_keys),
                        "hardware_vocab_size": len(hardware_vocab),
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "num_decoder_layers": NUM_DECODER_LAYERS,
                        "dim_feedforward": DIM_FEEDFORWARD,
                        "feature_mlp_layers": FEATURE_MLP_LAYERS,
                        "max_seq_len": MAX_PASS_SEQ_LEN,
                        "dropout": DROPOUT,
                        "context_tokens": CONTEXT_TOKENS,
                        "next_token_mode": args.next_token_mode,
                        "target_metric": TARGET_METRICS[
                            0
                        ],  # Add target metric to config
                    },
                },
                output_path / f"passgen_transformer_{TARGET_METRICS[0]}.pth",
            )

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_path / 'passgen_transformer_best.pth'}")


if __name__ == "__main__":
    main()
