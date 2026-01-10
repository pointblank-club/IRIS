import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path
import joblib
import numpy as np

# Import necessary components
from train_passformer_seqgen import PassGenTransformer, MAX_PASS_SEQ_LEN, TARGET_METRICS, CONTEXT_TOKENS, build_allowed_token_mask
from data_preprocessing_hybrid import load_and_preprocess_data

def calculate_simple_overlap(pred_seq, target_seq):
    """Calculate simple token overlap between sequences."""
    if not target_seq:
        return 0.0
    pred_set = set(pred_seq)
    target_set = set(target_seq)
    overlap = len(pred_set.intersection(target_set))
    return overlap / len(target_set)

def load_model_and_artifacts(model_path, preprocessing_output_path):
    """Load trained model and preprocessing artifacts."""
    print(f"Loading model from {model_path}...")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_config = checkpoint['config']

    # Load vocabularies and scalers
    with open(preprocessing_output_path / "joint_pass_vocab.json", 'r') as f:
        joint_pass_vocab = json.load(f)
    with open(preprocessing_output_path / "hardware_vocab.json", 'r') as f:
        hardware_vocab = json.load(f)
    with open(preprocessing_output_path / "feature_keys.json", 'r') as f:
        feature_keys = json.load(f)
    
    feature_scaler = joblib.load(preprocessing_output_path / "feature_scaler.pkl")
    target_metric_scaler = joblib.load(preprocessing_output_path / "target_metric_scaler.pkl")

    # Initialize model with config
    model = PassGenTransformer(
        vocab_size=model_config['vocab_size'],
        num_features=model_config['num_features'],
        hardware_vocab_size=model_config['hardware_vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        feature_mlp_layers=model_config['feature_mlp_layers'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout'],
        context_tokens=model_config.get('context_tokens', CONTEXT_TOKENS)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")

    return model, joint_pass_vocab, hardware_vocab, feature_scaler, target_metric_scaler, feature_keys

def tokens_to_passes(token_ids, id_to_pass, special_tokens):
    """Convert token IDs to pass names, filtering out special tokens."""
    return [id_to_pass.get(token_id, '<unk>') 
            for token_id in token_ids 
            if token_id not in special_tokens]

def main():
    parser = argparse.ArgumentParser(description="Generate compiler pass sequences and predict metrics.")
    parser.add_argument("--model_path", type=str, default="models_seqgen/passgen_transformer_best.pth",
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--preprocessing_output_dir", type=str, default="preprocessing_output",
                        help="Directory containing preprocessing artifacts.")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input flattened JSON dataset for sampling.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate and evaluate.")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated indices of specific samples to test (e.g., '0,5,10').")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search decoding (>=1).")
    parser.add_argument("--decode", type=str, choices=["greedy","beam"], default="beam", help="Decoding strategy.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sample details; only show summary.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    preprocessing_output_path = Path(args.preprocessing_output_dir)

    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        # Try alternative path
        alt_path = model_path.parent / "passgen_transformer_model_final.pth"
        if alt_path.exists():
            print(f"Found alternative model at {alt_path}")
            model_path = alt_path
        else:
            return

    # Load model and artifacts
    print("\nLoading model and preprocessing artifacts...")
    model, joint_pass_vocab, hardware_vocab, feature_scaler, target_metric_scaler, feature_keys = \
        load_model_and_artifacts(model_path, preprocessing_output_path)
    model.to(device)

    # Load and preprocess data
    print(f"\nLoading data from {args.input_json}...")
    processed_samples, _, _, _, _, _ = load_and_preprocess_data(
        args.input_json, MAX_PASS_SEQ_LEN, TARGET_METRICS, scale=False
    )
    print(f"Loaded {len(processed_samples)} samples")

    # Inverse vocabularies for display
    id_to_pass = {v: k for k, v in joint_pass_vocab.items()}
    id_to_hardware = {v: k for k, v in hardware_vocab.items()}
    special_tokens = [joint_pass_vocab["<pad>"], joint_pass_vocab["<sos>"], joint_pass_vocab["<eos>"]]

    # Determine which samples to evaluate
    if args.sample_indices:
        sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
        sample_indices = [idx for idx in sample_indices if 0 <= idx < len(processed_samples)]
    else:
        sample_indices = np.random.choice(len(processed_samples), 
                                         min(args.num_samples, len(processed_samples)), 
                                         replace=False).tolist()

    print(f"\n{'='*80}")
    print(f"Evaluating {len(sample_indices)} samples")
    print(f"{'='*80}")

    total_overlap = 0.0
    total_exact_matches = 0
    total_runtime_error = 0.0
    total_binary_error = 0.0

    for sample_num, sample_idx in enumerate(sample_indices, 1):
        sample = processed_samples[sample_idx]

        # Extract sample data
        raw_features = np.array(sample['program_features'], dtype=np.float32)
        scaled_features = feature_scaler.transform(raw_features.reshape(1, -1))[0]
        program_features = torch.tensor(scaled_features, dtype=torch.float32, device=device).unsqueeze(0)

        hardware_id = sample['hardware_id'].unsqueeze(0).to(device)
        target_sequence = sample['target_sequence']

        raw_target_labels = np.array(sample['labels'], dtype=np.float32)
        scaled_labels = target_metric_scaler.transform(raw_target_labels.reshape(1, -1))[0]
        target_labels = torch.tensor(scaled_labels, dtype=torch.float32)

        # Generate sequence
        with torch.no_grad():
            allowed_mask = build_allowed_token_mask(hardware_id, joint_pass_vocab, hardware_vocab, device)
            if args.decode == "beam" and args.beam_size > 1:
                generated_sequence = model.generate_sequence_beam(
                    program_features,
                    hardware_id,
                    joint_pass_vocab["<sos>"],
                    joint_pass_vocab["<eos>"],
                    joint_pass_vocab["<pad>"],
                    device,
                    MAX_PASS_SEQ_LEN,
                    beam_size=args.beam_size,
                    allowed_token_mask=allowed_mask
                )
            else:
                generated_sequence = model.generate_sequence_greedy(
                    program_features,
                    hardware_id,
                    joint_pass_vocab["<sos>"],
                    joint_pass_vocab["<eos>"],
                    joint_pass_vocab["<pad>"],
                    device,
                    MAX_PASS_SEQ_LEN,
                    allowed_token_mask=allowed_mask
                )
            generated_sequence = generated_sequence.squeeze(0)  # Remove batch dimension
            
            # Get predicted metrics (from regression head)
            # Use a minimal input sequence for forward pass
            initial_input = torch.tensor([[joint_pass_vocab["<sos>"]]], dtype=torch.long, device=device)
            _, predicted_metrics = model(program_features, hardware_id, initial_input)

        # Convert to CPU and numpy for processing
        generated_ids = generated_sequence.cpu().tolist()
        target_ids = target_sequence.cpu().tolist()
        predicted_metrics_np = predicted_metrics.cpu().numpy()
        target_labels_np = target_labels.cpu().numpy()

        # Unscale metrics
        predicted_metrics_unscaled = target_metric_scaler.inverse_transform(predicted_metrics_np)
        target_labels_unscaled = target_metric_scaler.inverse_transform(target_labels_np.reshape(1, -1))

        # Convert token IDs to pass names
        generated_passes = tokens_to_passes(generated_ids, id_to_pass, special_tokens)
        target_passes = tokens_to_passes(target_ids, id_to_pass, special_tokens)
        
        # Calculate metrics
        overlap = calculate_simple_overlap(generated_passes, target_passes)
        exact_match = 1 if generated_passes == target_passes else 0
        
        runtime_pred = predicted_metrics_unscaled[0, 0]
        binary_pred = predicted_metrics_unscaled[0, 1]
        runtime_true = target_labels_unscaled[0, 0]
        binary_true = target_labels_unscaled[0, 1]
        
        runtime_error = abs(runtime_pred - runtime_true)
        binary_error = abs(binary_pred - binary_true)
        
        # Accumulate for averages
        total_overlap += overlap
        total_exact_matches += exact_match
        total_runtime_error += runtime_error
        total_binary_error += binary_error

        # Display results
        hardware_name = id_to_hardware.get(hardware_id.item(), "unknown")
        
        if not args.quiet:
            print(f"\n{'-'*80}")
            print(f"Sample {sample_num} (Index: {sample_idx})")
            print(f"{'-'*80}")
            print(f"Hardware: {hardware_name}")
            print(f"\nGenerated Sequence ({len(generated_passes)} passes):")
            print(f"  {' -> '.join(generated_passes[:10])}{'...' if len(generated_passes) > 10 else ''}")
            print(f"\nTarget Sequence ({len(target_passes)} passes):")
            print(f"  {' -> '.join(target_passes[:10])}{'...' if len(target_passes) > 10 else ''}")
            print(f"\nSequence Metrics:")
            print(f"  Token Overlap: {overlap:.2%}")
            print(f"  Exact Match: {'Yes' if exact_match else 'No'}")
            print(f"\nPredicted Performance Metrics:")
            print(f"  Runtime:     {runtime_pred:>10.4f}  (True: {runtime_true:>10.4f}, Error: {runtime_error:.4f})")
            print(f"  Binary Size: {binary_pred:>10.4f}  (True: {binary_true:>10.4f}, Error: {binary_error:.4f})")

    # Print summary statistics
    num_samples_eval = len(sample_indices)
    print(f"\n{'='*80}")
    print(f"Summary Statistics (over {num_samples_eval} samples)")
    print(f"{'='*80}")
    print(f"Average Token Overlap:     {total_overlap / num_samples_eval:.2%}")
    print(f"Exact Match Rate:          {total_exact_matches / num_samples_eval:.2%}")
    print(f"Average Runtime MAE:       {total_runtime_error / num_samples_eval:.4f}")
    print(f"Average Binary Size MAE:   {total_binary_error / num_samples_eval:.4f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
