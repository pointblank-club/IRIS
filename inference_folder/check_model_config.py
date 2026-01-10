#!/usr/bin/env python3
"""Check what's in the transformer model checkpoint."""

import torch
from pathlib import Path

model_path = Path("models_seqgen/passgen_transformer_model_best.pth")

if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("Checkpoint keys:", checkpoint.keys())
    print("\nModel config:")
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            print(f"  {key}: {value}")
    
    print("\nState dict keys (first 20):")
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    for key in state_dict_keys[:20]:
        print(f"  {key}")
    
    if len(state_dict_keys) > 20:
        print(f"  ... and {len(state_dict_keys) - 20} more keys")
else:
    print("Model file not found!")
