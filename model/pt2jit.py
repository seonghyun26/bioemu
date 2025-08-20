#!/usr/bin/env python3
"""
Script to load a trained MLCV model, add postprocessing, and save as JIT traced model.
Usage: python pt2jit.py <date> <molecule>
"""

import sys
import os
import lightning
import torch
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from mlcolvar.core.transform import Statistics, Transform
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization


class DIM_NORMALIZATION(Transform):
    """Dimension normalization transform for MLCV model."""
    def __init__(self, feature_dim=1):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
        
    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

class MLCV(BaseCV, lightning.LightningModule):
    """MLCV model class for our method."""
    BLOCKS = ["norm_in", "encoder",]

    def __init__(self, mlcv_dim: int, encoder_layers: list, dim_normalization: bool = False, options: dict = None, **kwargs):
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)
        # ======= OPTIONS =======
        options = self.parse_options(options)
        
        # ======= BLOCKS =======
        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])
        if dim_normalization:
            self.postprocessing = DIM_NORMALIZATION(mlcv_dim)

def sanitize_range(range_tensor: torch.Tensor) -> torch.Tensor:
    """Sanitize range tensor to avoid division by zero."""
    if (range_tensor < 1e-6).nonzero().sum() > 0:
        print(
            "[Warning] Normalization: the following features have a range of values < 1e-6:",
            (range_tensor < 1e-6).nonzero(),
        )
    range_tensor[range_tensor < 1e-6] = 1.0
    return range_tensor


class PostProcess(Transform):
    """Post-processing module for MLCV normalization and sign flipping"""
    def __init__(
        self,
        stats=None,
        reference_frame_cv=None,
        feature_dim=1,
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("range", torch.ones(feature_dim))
        
        if stats is not None:
            min_val = stats["min"]
            max_val = stats["max"]
            self.mean = (max_val + min_val) / 2.0
            range_val = (max_val - min_val) / 2.0
            self.range = sanitize_range(range_val)
        
        if reference_frame_cv is not None:
            self.register_buffer(
                "flip_sign",
                torch.ones(1) * -1 if reference_frame_cv < 0 else torch.ones(1)
            )
        else:
            self.register_buffer("flip_sign", torch.ones(1))
        
    def forward(self, x):
        x = x.sub(self.mean).div(self.range)
        x = x * self.flip_sign
        return x


def load_molecule_data(molecule: str, device: torch.device = torch.device('cpu')):
    """Load molecule data for statistics computation."""
    simulation_idx = 0
    data_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-cad.pt"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Molecule data not found at {data_path}")
    
    print(f"Loading molecule data from {data_path}")
    data = torch.load(data_path, map_location=device)
    print(f"Data shape: {data.shape}")
    return data


def add_postprocessing_to_model(model, projection_data, device):
    """Add postprocessing module to the model."""
    print("Computing postprocessing statistics...")
    
    # Evaluate model on full dataset
    model.eval()
    with torch.no_grad():
        cv = model(projection_data)
    
    print(f"CV shape: {cv.shape}")
    print(f"CV range: [{cv.min():.6f}, {cv.max():.6f}]")
    
    # Compute statistics for post-processing
    stats = Statistics(cv.cpu()).to_dict()
    
    # Create and attach post-processing module
    mlcv_dim = cv.shape[1]
    postprocessing = PostProcess(
        stats=stats,
        reference_frame_cv=None,  # No reference frame for sign flipping
        feature_dim=mlcv_dim
    ).to(device)
    
    # Attach to model
    model.postprocessing = postprocessing
    
    # Test post-processed output
    with torch.no_grad():
        postprocessed_cv = model(projection_data)
        print(f"Post-processed CV range: [{postprocessed_cv.min():.6f}, {postprocessed_cv.max():.6f}]")
    
    print("Post-processing module attached successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Convert MLCV model to JIT with postprocessing')
    parser.add_argument('date', type=str, help='Date string for model path')
    parser.add_argument('molecule', type=str, help='Molecule name (e.g., CLN025, 2JOF)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    
    # Load model
    model_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/{args.date}/mlcv_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if args.molecule == "2JOF":
        INPUT_DIM = 190
    elif args.molecule == "CLN025":
        INPUT_DIM = 45 
    else:
        raise ValueError(f"Molecule {args.molecule} not supported")
    MLCV_DIM = 2
    
    print(f"Loading model from {model_path}")
    model_state = torch.load(model_path)
    mlcv_state_dict = model_state["mlcv_state_dict"]
    encoder_layers = [INPUT_DIM, 100, 100, MLCV_DIM]
    options = {
        "encoder": {
            "activation": "tanh",
            "dropout": [0.1, 0.1, 0.1]
        },
        "norm_in": {},
    }
    model = MLCV(
        mlcv_dim=MLCV_DIM,
        encoder_layers=encoder_layers,
        dim_normalization=False,
        options=options
    )
    model.load_state_dict(mlcv_state_dict)
    model.eval()
    
    # Load molecule data
    projection_data = load_molecule_data(args.molecule, device)
    model = add_postprocessing_to_model(model, projection_data, device)
    
    # Save JIT traced model
    output_path = model_path.replace('.pt', '-jit.pt')
    print(f"Saving JIT model to {output_path}")
    model.trainer = Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
    dummy_input = torch.randn(1, projection_data.shape[1]).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    
    print(f"Successfully saved JIT model with postprocessing to {output_path}")


if __name__ == "__main__":
    main()
