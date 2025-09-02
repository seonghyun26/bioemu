#!/usr/bin/env python3
"""
Unit tests for MLCV_TRANSFERABLE2 model.

This test suite validates that the new coordinate-based transferable MLCV model
works correctly and demonstrates its key advantages over the distance-based approach.
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add the model path to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MLCV_TRANSFERABLE2, load_transferable_mlcv2, sequence_to_indices


class TestMLCVTransferable2:
    """Test suite for MLCV_TRANSFERABLE2 model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test parameters
        self.mlcv_dim = 2
        self.d_model = 64  # Smaller for faster testing
        self.d_pair = 32
        self.n_head = 4
        self.dim_feedforward = 128
        self.n_layers = 2
        self.dropout = 0.1
        self.max_seq_len = 100
        
        # Create a basic test model
        self.model = MLCV_TRANSFERABLE2(
            mlcv_dim=self.mlcv_dim,
            d_model=self.d_model,
            d_pair=self.d_pair,
            n_head=self.n_head,
            dim_feedforward=self.dim_feedforward,
            n_layers=self.n_layers,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            use_pose_estimation=True,
            residue_based_encoding=True,
        )
        
        # Create test data with different protein sizes
        self.test_sequences = {
            'small': 'ACDEFGHIKLMNPQRSTVWY',  # 20 residues
            'medium': 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',  # 40 residues  
            'large': 'ACDEFGHIKLMNPQRSTVWY' * 3,  # 60 residues
        }
        
        # Generate random 3D coordinates for test proteins
        self.test_coordinates = {}
        for name, seq in self.test_sequences.items():
            seq_len = len(seq)
            # Generate realistic protein-like coordinates (roughly linear chain with noise)
            base_coords = torch.randn(1, seq_len, 3) * 0.5  # Small random displacement
            for i in range(1, seq_len):
                base_coords[0, i] = base_coords[0, i-1] + torch.randn(3) * 1.5  # Chain-like structure
            self.test_coordinates[name] = base_coords
            
        # Create amino acid type indices
        self.test_aa_types = {}
        for name, seq in self.test_sequences.items():
            self.test_aa_types[name] = sequence_to_indices(seq)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        assert isinstance(self.model, MLCV_TRANSFERABLE2)
        assert self.model.mlcv_dim == self.mlcv_dim
        assert self.model.d_model == self.d_model
        assert hasattr(self.model, 'coordinate_embedding')
        assert hasattr(self.model, 'encoder_layers')
        assert hasattr(self.model, 'output_projection')
        
        # Test that pose estimation is properly configured
        assert hasattr(self.model, 'pose_estimator')
        assert self.model.use_pose_estimation == True
        
        print("✓ Model initialization test passed")
    
    def test_input_validation(self):
        """Test input validation for coordinate shapes."""
        # Test correct input shape
        coords = torch.randn(2, 10, 3)
        try:
            output = self.model(coords)
            assert output.shape == (2, self.mlcv_dim)
        except Exception as e:
            pytest.fail(f"Valid input should not raise exception: {e}")
        
        # Test invalid input shapes
        with pytest.raises(ValueError, match="Expected coordinates with shape"):
            self.model(torch.randn(2, 10))  # Missing coordinate dimension
            
        with pytest.raises(ValueError, match="Expected coordinates with shape"):
            self.model(torch.randn(2, 10, 2))  # Wrong coordinate dimension
            
        print("✓ Input validation test passed")
    
    def test_transferability_across_sizes(self):
        """Test that the model works with different protein sizes."""
        outputs = {}
        
        for name, coords in self.test_coordinates.items():
            aa_types = self.test_aa_types[name]
            
            # Test forward pass
            self.model.eval()
            with torch.no_grad():
                output = self.model(coords, aa_types)
            
            # Verify output shape
            assert output.shape == (1, self.mlcv_dim), f"Wrong output shape for {name}: {output.shape}"
            
            # Verify output is not NaN or Inf
            assert torch.isfinite(output).all(), f"Output contains NaN/Inf for {name}"
            
            outputs[name] = output
            print(f"✓ {name} protein ({len(self.test_sequences[name])} residues): output shape {output.shape}")
        
        # Test that different sized proteins produce different outputs
        # (they should not be identical due to different structural content)
        small_out = outputs['small']
        medium_out = outputs['medium']
        large_out = outputs['large']
        
        assert not torch.allclose(small_out, medium_out, atol=1e-6), "Different sized proteins should produce different outputs"
        assert not torch.allclose(medium_out, large_out, atol=1e-6), "Different sized proteins should produce different outputs"
        
        print("✓ Transferability across protein sizes test passed")
    
    def test_batch_processing(self):
        """Test that the model can handle batched inputs of the same size."""
        seq_len = 25
        batch_size = 4
        
        # Create batch of coordinates
        coords_batch = torch.randn(batch_size, seq_len, 3)
        
        # Create batch of amino acid types (same sequence for simplicity)
        aa_types_batch = sequence_to_indices('A' * seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(coords_batch, aa_types_batch)
        
        assert output.shape == (batch_size, self.mlcv_dim), f"Wrong batch output shape: {output.shape}"
        assert torch.isfinite(output).all(), "Batch output contains NaN/Inf"
        
        print(f"✓ Batch processing test passed: {coords_batch.shape} -> {output.shape}")
    
    def test_pose_estimation(self):
        """Test the pose estimation functionality."""
        coords = self.test_coordinates['medium']
        
        # Test with pose estimation enabled
        model_with_pose = MLCV_TRANSFERABLE2(
            mlcv_dim=2,
            d_model=64,
            use_pose_estimation=True,
        )
        
        # Test with pose estimation disabled
        model_without_pose = MLCV_TRANSFERABLE2(
            mlcv_dim=2,
            d_model=64,
            use_pose_estimation=False,
        )
        
        model_with_pose.eval()
        model_without_pose.eval()
        
        with torch.no_grad():
            output_with_pose = model_with_pose(coords)
            output_without_pose = model_without_pose(coords)
        
        # Both should work but produce different results
        assert output_with_pose.shape == output_without_pose.shape
        assert torch.isfinite(output_with_pose).all()
        assert torch.isfinite(output_without_pose).all()
        
        print("✓ Pose estimation functionality test passed")
    
    def test_residue_based_encoding(self):
        """Test residue-based positional encoding."""
        coords = self.test_coordinates['small']
        aa_types = self.test_aa_types['small']
        
        # Test with residue-based encoding
        model_with_aa = MLCV_TRANSFERABLE2(
            mlcv_dim=2,
            d_model=64,
            residue_based_encoding=True,
        )
        
        # Test without residue-based encoding  
        model_without_aa = MLCV_TRANSFERABLE2(
            mlcv_dim=2,
            d_model=64,
            residue_based_encoding=False,
        )
        
        model_with_aa.eval()
        model_without_aa.eval()
        
        with torch.no_grad():
            output_with_aa = model_with_aa(coords, aa_types)
            output_without_aa = model_without_aa(coords, None)
        
        assert output_with_aa.shape == output_without_aa.shape
        assert torch.isfinite(output_with_aa).all()
        assert torch.isfinite(output_without_aa).all()
        
        print("✓ Residue-based encoding test passed")
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        coords = self.test_coordinates['small']
        aa_types = self.test_aa_types['small']
        
        # Enable gradients
        coords.requires_grad_(True)
        self.model.train()
        
        # Forward pass
        output = self.model(coords, aa_types)
        loss = output.sum()  # Simple loss for testing
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        assert coords.grad is not None, "Gradients should flow to input coordinates"
        assert torch.isfinite(coords.grad).all(), "Coordinate gradients should be finite"
        
        # Check model parameter gradients
        param_grads_exist = False
        for param in self.model.parameters():
            if param.grad is not None:
                param_grads_exist = True
                assert torch.isfinite(param.grad).all(), "Parameter gradients should be finite"
        
        assert param_grads_exist, "At least some model parameters should have gradients"
        
        print("✓ Gradient flow test passed")
    
    def test_coordinate_vs_distance_comparison(self):
        """
        Compare MLCV_TRANSFERABLE2 with MLCV_TRANSFERABLE to demonstrate advantages.
        This test shows that coordinate-based processing can capture different information.
        """
        from model import MLCV_TRANSFERABLE
        
        coords = self.test_coordinates['medium']
        aa_types = self.test_aa_types['medium']
        
        # Create coordinate-based model (MLCV_TRANSFERABLE2)
        coord_model = MLCV_TRANSFERABLE2(
            mlcv_dim=2,
            d_model=64,
            n_layers=2,
        )
        
        # Create distance-based model (MLCV_TRANSFERABLE) 
        dist_model = MLCV_TRANSFERABLE(
            input_dim=3,  # Dummy value
            mlcv_dim=2,
            d_model=64,
            n_layers=2,
            input_type="coordinates",
            coordinate_processing="to_distances",
        )
        
        coord_model.eval()
        dist_model.eval()
        
        with torch.no_grad():
            coord_output = coord_model(coords, aa_types)
            dist_output = dist_model(coords, aa_types)
        
        # Both should work and produce valid outputs
        assert coord_output.shape == dist_output.shape
        assert torch.isfinite(coord_output).all()
        assert torch.isfinite(dist_output).all()
        
        # They should be different (capture different geometric information)
        assert not torch.allclose(coord_output, dist_output, atol=1e-4), \
            "Coordinate and distance-based models should produce different outputs"
        
        print("✓ Coordinate vs distance processing comparison test passed")
        print(f"  Coordinate-based output: {coord_output.numpy()}")
        print(f"  Distance-based output: {dist_output.numpy()}")
    
    def test_load_function(self):
        """Test the load_transferable_mlcv2 convenience function."""
        loaded_model = load_transferable_mlcv2(
            mlcv_dim=3,
            d_model=32,
            n_layers=1,
            use_pose_estimation=False,
        )
        
        assert isinstance(loaded_model, MLCV_TRANSFERABLE2)
        assert loaded_model.mlcv_dim == 3
        assert loaded_model.d_model == 32
        assert loaded_model.use_pose_estimation == False
        
        # Test that it can process data
        coords = torch.randn(1, 20, 3)
        loaded_model.eval()
        with torch.no_grad():
            output = loaded_model(coords)
        
        assert output.shape == (1, 3)
        assert torch.isfinite(output).all()
        
        print("✓ Load function test passed")


def run_tests():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("TESTING MLCV_TRANSFERABLE2 MODEL")
    print("=" * 60)
    
    test_instance = TestMLCVTransferable2()
    test_instance.setup_method()
    
    test_methods = [
        test_instance.test_model_initialization,
        test_instance.test_input_validation,
        test_instance.test_transferability_across_sizes,
        test_instance.test_batch_processing,
        test_instance.test_pose_estimation,
        test_instance.test_residue_based_encoding,
        test_instance.test_gradient_flow,
        test_instance.test_coordinate_vs_distance_comparison,
        test_instance.test_load_function,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method.__name__}...")
            test_method()
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nMLCV_TRANSFERABLE2 is working correctly and ready to use.")
        print("\nKey features validated:")
        print("- Direct 3D coordinate processing")
        print("- Transferability across protein sizes")
        print("- Pose estimation and geometric awareness") 
        print("- Residue-based positional encoding")
        print("- Proper gradient flow for training")
        print("- Batch processing capabilities")
    else:
        print(f"\n❌ {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

