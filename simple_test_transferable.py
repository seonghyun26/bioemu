#!/usr/bin/env python3
"""
Simple test for MLCV_TRANSFERABLE without full environment dependencies.
"""

import torch
import math


class PositionalEncodingMatrix(torch.nn.Module):
    """Positional encoding for pairwise distance matrices."""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix [max_len, max_len, d_model]
        pe = torch.zeros(max_len, max_len, d_model)
        
        for i in range(max_len):
            for j in range(max_len):
                pos_i = i
                pos_j = j
                rel_dist = abs(i - j)
                
                # Use sinusoidal encoding similar to transformer positional encoding
                for k in range(0, d_model, 4):
                    if k < d_model:
                        pe[i, j, k] = math.sin(pos_i / (10000 ** (k / d_model)))
                    if k + 1 < d_model:
                        pe[i, j, k + 1] = math.cos(pos_i / (10000 ** (k / d_model)))
                    if k + 2 < d_model:
                        pe[i, j, k + 2] = math.sin(pos_j / (10000 ** ((k + 2) / d_model)))
                    if k + 3 < d_model:
                        pe[i, j, k + 3] = math.cos(rel_dist / (10000 ** ((k + 3) / d_model)))
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len, :seq_len, :]


class SimpleTransferableMLCV(torch.nn.Module):
    """Simplified transferable MLCV for testing core functionality."""
    
    def __init__(self, mlcv_dim: int = 2, d_model: int = 128, max_seq_len: int = 100):
        super().__init__()
        self.mlcv_dim = mlcv_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Distance matrix preprocessing
        self.distance_embedding = torch.nn.Linear(1, d_model)
        
        # Position encoding for distance matrix indices
        self.pos_encoding = PositionalEncodingMatrix(d_model, max_seq_len)
        
        # Simple encoder (without full SA layers for testing)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU()
        )
        
        # Global pooling and output projection
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, mlcv_dim)
        )
    
    def cad_to_sequence_representation(self, cad_distances: torch.Tensor):
        """Convert pairwise CA distances to sequence representation."""
        batch_size = cad_distances.shape[0]
        n_pairs = cad_distances.shape[1]
        
        # Reconstruct sequence length from number of pairs
        n_residues = int((1 + math.sqrt(1 + 8 * n_pairs)) / 2)
        
        # Embed distance values
        dist_embedded = self.distance_embedding(cad_distances.unsqueeze(-1))
        
        # Create pairwise distance matrix [B, L, L, d_model]
        distance_matrix = torch.zeros(batch_size, n_residues, n_residues, self.d_model, 
                                    device=cad_distances.device, dtype=cad_distances.dtype)
        
        # Fill upper triangular part
        idx = 0
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distance_matrix[:, i, j] = dist_embedded[:, idx]
                distance_matrix[:, j, i] = dist_embedded[:, idx]  # Symmetric
                idx += 1
        
        # Add positional encoding
        pos_encoded = self.pos_encoding(n_residues)
        distance_matrix = distance_matrix + pos_encoded.unsqueeze(0)
        
        # Create sequence representation by averaging over pairs
        x1d = distance_matrix.mean(dim=2)  # [B, L, d_model]
        
        return x1d
    
    def forward(self, cad_distances: torch.Tensor) -> torch.Tensor:
        """Forward pass of simplified transferable MLCV."""
        # Convert distances to sequence representation
        x1d = self.cad_to_sequence_representation(cad_distances)
        
        # Apply encoder
        x1d = self.encoder(x1d)
        
        # Global pooling over sequence length
        pooled = self.global_pool(x1d.transpose(1, 2)).squeeze(-1)
        
        # Output projection
        cv = self.output_projection(pooled)
        
        return cv


def test_transferable_functionality():
    """Test core transferable functionality."""
    
    print("Testing Simple Transferable MLCV")
    print("=" * 50)
    
    # Create model
    model = SimpleTransferableMLCV(mlcv_dim=2, d_model=64, max_seq_len=100)
    model.eval()
    
    # Test different protein sizes
    protein_sizes = [10, 20, 35]  # CLN025=10, etc.
    batch_size = 3
    
    print(f"Batch size: {batch_size}")
    print(f"Output dimension: {model.mlcv_dim}")
    print()
    
    for n_residues in protein_sizes:
        # Calculate number of pairwise distances
        n_pairs = n_residues * (n_residues - 1) // 2
        
        print(f"Protein: {n_residues} residues, {n_pairs} pairs")
        
        # Create dummy data
        cad_distances = torch.randn(batch_size, n_pairs) * 2 + 8
        cad_distances = torch.abs(cad_distances)
        
        # Forward pass
        with torch.no_grad():
            output = model(cad_distances)
        
        print(f"  Input shape:  {cad_distances.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Verify output shape
        expected_shape = (batch_size, model.mlcv_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"  ✓ Correct output shape")
        
        print()
    
    print("✓ All tests passed!")
    print("✓ Model handles variable protein sizes correctly")
    print("✓ Output dimension is consistent regardless of input size")
    
    # Test gradient computation
    print("\nTesting gradients...")
    cad_test = torch.randn(2, 45, requires_grad=True)  # CLN025 size
    output = model(cad_test)
    loss = output.sum()
    loss.backward()
    
    if cad_test.grad is not None:
        print(f"✓ Gradients computed successfully")
        print(f"  Gradient norm: {cad_test.grad.norm().item():.6f}")
    else:
        print("❌ No gradients computed")
    
    return True


if __name__ == "__main__":
    test_transferable_functionality()
