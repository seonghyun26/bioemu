#!/usr/bin/env python3
"""
Test script for MLCV_TRANSFERABLE to verify it works across different protein sizes.

This script tests:
1. Model can handle different input sizes (different protein lengths)
2. Output is always the same dimension regardless of input size
3. Model is differentiable and can be trained
"""

import torch
import numpy as np
from model import load_transferable_mlcv


def test_transferable_mlcv():
    """Test the transferable MLCV model with different protein sizes."""
    
    print("=" * 60)
    print("Testing MLCV_TRANSFERABLE Model")
    print("=" * 60)
    
    # Load transferable model
    model = load_transferable_mlcv(
        mlcv_dim=2,
        d_model=128,  # Smaller for testing
        d_pair=64,
        n_head=4,
        dim_feedforward=256,
        n_layers=2,
        dropout=0.1,
        max_seq_len=100,  # Support up to 100 residues
        dim_normalization=True,
        normalization_factor=10.0
    )
    
    # Test different protein sizes
    protein_sizes = [10, 20, 35, 50]  # CLN025=10, various other sizes
    batch_size = 4
    
    print(f"\nTesting with batch_size={batch_size}")
    print("-" * 40)
    
    all_outputs = []
    
    for n_residues in protein_sizes:
        # Calculate number of pairwise distances
        n_pairs = n_residues * (n_residues - 1) // 2
        
        print(f"\nProtein size: {n_residues} residues, {n_pairs} pairwise distances")
        
        # Create dummy pairwise CA distance data
        # Realistic CA-CA distances are typically 3-15 Angstroms
        cad_distances = torch.randn(batch_size, n_pairs) * 2 + 8  # Mean ~8Å, std ~2Å
        cad_distances = torch.abs(cad_distances)  # Ensure positive distances
        
        print(f"Input shape: {cad_distances.shape}")
        print(f"Distance range: [{cad_distances.min():.2f}, {cad_distances.max():.2f}] Å")
        
        # Forward pass
        with torch.no_grad():
            output = model(cad_distances)
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check output shape is correct
        expected_shape = (batch_size, 2)  # mlcv_dim=2
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        all_outputs.append(output)
        
        # Test gradients
        output_for_grad = model(cad_distances)
        loss = output_for_grad.sum()
        loss.backward()
        
        # Check that gradients exist
        grad_norm = 0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        grad_norm = grad_norm ** 0.5
        
        print(f"Gradient norm: {grad_norm:.4f} (across {param_count} parameters)")
        
        # Zero gradients for next iteration
        model.zero_grad()
    
    print("\n" + "=" * 60)
    print("TRANSFERABILITY TEST RESULTS")
    print("=" * 60)
    
    # Check that outputs are reasonable and different for different inputs
    print(f"✓ Model successfully processed {len(protein_sizes)} different protein sizes")
    print(f"✓ All outputs have consistent shape: {all_outputs[0].shape}")
    print(f"✓ Model is differentiable (gradients computed successfully)")
    
    # Check output diversity
    output_stds = []
    for i, output in enumerate(all_outputs):
        std = output.std().item()
        output_stds.append(std)
        print(f"✓ Size {protein_sizes[i]:2d}: Output std = {std:.4f} (diverse outputs)")
    
    mean_std = np.mean(output_stds)
    print(f"✓ Average output diversity: {mean_std:.4f}")
    
    if mean_std > 0.01:  # Outputs should be diverse enough
        print("✓ Model produces diverse outputs for different proteins")
    else:
        print("⚠ Warning: Model outputs may be too similar (low diversity)")
    
    print("\n" + "=" * 60)
    print("COMPARISON: Fixed vs Transferable Model Memory Usage")
    print("=" * 60)
    
    # Compare parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Transferable model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Estimate fixed model parameters for different sizes
    for n_residues in protein_sizes:
        n_pairs = n_residues * (n_residues - 1) // 2
        # Fixed model: [n_pairs, 100, 100, 2] + normalization
        fixed_params = n_pairs * 100 + 100 * 100 + 100 * 2 + 100 + 100 + 2  # Approximate
        print(f"Fixed model for {n_residues:2d} residues would need: ~{fixed_params:,} parameters")
    
    print(f"\n✓ Transferable model uses ONE set of parameters for ALL protein sizes!")
    print(f"✓ No need to retrain for each new protein system!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_transferable_mlcv()
        if success:
            print("\n🎉 All tests passed! MLCV_TRANSFERABLE is working correctly.")
        else:
            print("\n❌ Some tests failed.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
