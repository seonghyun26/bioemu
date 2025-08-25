#!/usr/bin/env python3
"""
Script to load a trained MLCV model, add postprocessing, and save as JIT traced model.
Usage: python pt2jit.py <date> <molecule>
"""

import os
import lightning
import torch
import argparse
import math
import torch.nn as nn

from pytorch_lightning import Trainer

from mlcolvar.core.transform import Statistics, Transform
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization



class FeedForward(nn.Module):
    """Standard single hidden layer MLP with dropout and GELU activations."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)

class SAEncoderLayer(nn.Module):
    """IPA interleaved with layernorm and MLP."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SAAttention(d_model=d_model, d_pair=d_pair, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(
        self,
        x1d: torch.Tensor,
        x2d: torch.Tensor,
        pose: tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
    ) -> torch.Tensor:
        x1d = x1d + self.attn(self.norm1(x1d), x2d, pose, bias)
        x1d = x1d + self.ffn(self.norm2(x1d))
        return x1d
    
class SAAttention(nn.Module):
    """DiG version of the invariant point attention module. See AF2 supplement Alg 22.
    I believe SA might stand for "Structural Attention", see App B.3 in the DiG paper.

    The forward pass of this module is identical to IPA as described in Alg 22 in AF2 supplement,
    with the following changes:
        1. An extra linear map is applied to the pair representation.
        2. Dropout is applied to the output. (In AF2 it is applied outside of IPA. This may be
            equivalent.)


    Args:
        d_model: Dimension of attention dot product * number of heads.
        d_pair: Dimension of the pair representation.
        n_head: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_pair: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads.")
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.scalar_query = nn.Linear(d_model, d_model, bias=False)
        self.scalar_key = nn.Linear(d_model, d_model, bias=False)
        self.scalar_value = nn.Linear(d_model, d_model, bias=False)
        self.pair_bias = nn.Linear(d_pair, n_head, bias=False)
        self.point_query = nn.Linear(
            d_model, n_head * 3 * 4, bias=False
        )  # 4 is N_query_points in Alg 22.
        self.point_key = nn.Linear(
            d_model, n_head * 3 * 4, bias=False
        )  # 4 is N_query_points in Alg 22.
        self.point_value = nn.Linear(
            d_model, n_head * 3 * 8, bias=False
        )  # 8 is N_point_values in Alg 22.

        self.scalar_weight = 1.0 / math.sqrt(3 * self.d_k)  # Alg 22 line 7, w_L / sqrt(d_k).
        self.point_weight = 1.0 / math.sqrt(3 * 4 * 9 / 2)  # Alg 22 line 7, w_C * w_L.
        self.trained_point_weight = nn.Parameter(
            torch.rand(n_head)
        )  # gamma^h, AF2 Supp Section 1.8.2.
        self.pair_weight = 1.0 / math.sqrt(3)  # Alg 22 line 7, w_L.

        self.pair_value = nn.Linear(
            d_pair, d_model, bias=False
        )  # NOTE: AF2 IPA does not have this.

        self.fc_out = nn.Linear(d_model * 2 + n_head * 8 * 4, d_model, bias=True)  # Alg 22 line 11.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x1d: torch.Tensor,
        x2d: torch.Tensor,
        pose: tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
    ) -> torch.Tensor:

        """Forward pass of the SAAttention module.

        Args:
            x1d: Invariant sequence representation.
            x2d: Invariant pair representation.
            pose: Tuple of translation and inverse rotation vectors.
            bias: Pair bias, used to encode masking.
        """
        T, R = pose[0], pose[1].transpose(
            -1, -2
        )  # Transpose to go back to rotations from inverse rotations.

        # Compute scalar attention queries keys and values.
        # Alg 22 line 1, shape [B, L, nhead, C].
        q_scalar = self.scalar_query(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)
        k_scalar = self.scalar_key(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)
        v_scalar = self.scalar_value(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)

        # Perform scalar dot product attention.
        # Alg 22 line 7, shape [B, nhead, L, L]
        scalar_attn = torch.einsum("bihc,bjhc->bhij", q_scalar * self.scalar_weight, k_scalar)

        # Compute point attention queries keys and values.
        # Alg 22 line 2-3, shape [B, L, nhead, num_points, 3]
        q_point_local = self.point_query(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)
        k_point_local = self.point_key(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)
        v_point_local = self.point_value(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)

        def apply_affine(point: torch.Tensor, T: torch.Tensor, R: torch.Tensor):
            """Apply affine transformation (T, R) to point x. Acts as x -> R @ x + T. This follows
            AF2 Supplement Section 1.1.

            Args:
                point: Point to transform.
                T: Translation vector.
                R: Rotation matrix.

            Returns:
                Transformed point.
            """
            return (
                torch.matmul(R[:, :, None, None], point.unsqueeze(-1)).squeeze(-1)
                + T[:, :, None, None]
            )

        # Apply the frames to the attention points.
        # Alg 22 lines 7 and 10, shape [B, L, nhead, num_points, 3]
        q_point_global = apply_affine(q_point_local, T, R)
        k_point_global = apply_affine(k_point_local, T, R)
        v_point_global = apply_affine(v_point_local, T, R)

        # Compute squared distances between transformed points.
        # Alg 22 line 7, shape [B, L, L, nhead, num]
        point_attn = torch.norm(q_point_global.unsqueeze(2) - k_point_global.unsqueeze(1), dim=-1)
        point_weight = self.point_weight * F.softplus(
            self.trained_point_weight
        )  # w_L * w_C * gamma^h
        point_attn = (
            -0.5 * point_weight[:, None, None] * torch.sum(point_attn, dim=-1).permute(0, 3, 1, 2)
        )

        # Alg 22 line 4.
        pair_attn = self.pair_weight * self.pair_bias(x2d).permute(0, 3, 1, 2)

        # Compute attention logits, Alg 22 line 7.
        attn_logits = scalar_attn + point_attn + pair_attn + bias  # [B, nhead, L, L]

        # Compute attention weights.
        # Alg 22 line 7, shape [B, nhead, L, L]
        attn = torch.softmax(attn_logits, dim=-1)

        # Alg 22 line 9.
        out_scalar = torch.einsum("bhij,bjhc->bihc", attn, v_scalar)
        out_scalar = out_scalar.reshape(*out_scalar.shape[:2], -1)

        # Alg 22 line 10.
        with torch.amp.autocast("cuda", enabled=False):
            out_point_global = torch.einsum(
                "bhij,bjhcp->bihcp", attn.float(), v_point_global.float()
            )
        # Inverse affine transformation, as per Alg 22 line 10, and AF2 Supplement Section 1.1.
        out_point_local = torch.matmul(
            R.transpose(-1, -2)[:, :, None, None],
            (out_point_global - T[:, :, None, None]).unsqueeze(-1),
        ).squeeze(-1)

        # Alg 22 line 11.
        out_point_norm = torch.norm(out_point_local, dim=-1)
        out_point_norm = out_point_norm.reshape(*out_point_norm.shape[:2], -1)
        out_point_local = out_point_local.reshape(*out_point_local.shape[:2], -1)

        # NOTE: AF2 IPA does not project x2d as in here, i.e., v_pair = x2d in AF2.
        v_pair = self.pair_value(x2d).reshape(*x2d.shape[:-1], self.n_head, -1)

        # Alg 22 line 8.
        out_pair = torch.einsum("bhij,bijhc->bihc", attn, v_pair)
        out_pair = out_pair.reshape(*out_pair.shape[:2], -1)

        # Alg 22 line 11.
        out_feat = torch.cat([out_scalar, out_point_local, out_pair, out_point_norm], dim=-1)

        # NOTE: AF2 includes dropout outside IPA, not inside. See AF2 Alg 22 line 6.
        x = self.dropout(self.fc_out(out_feat))
        return x  # [B, L, C]


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


class MLCV_TRANSFERABLE(BaseCV, lightning.LightningModule):
    """
    Transferable MLCV model that can handle variable protein sizes.
    
    This model uses a structure-aware attention mechanism (SAEncoderLayer) 
    to process either pairwise distance matrices or 3D coordinates of different 
    sizes and output fixed-size collective variables, making it transferable 
    across proteins.
    
    Key features:
    - Handles variable input sizes (different protein lengths)
    - Supports both pairwise distances [B, n_pairs] and 3D coordinates [B, N, 3] as input
    - For coordinates: can either convert to distances first or process directly
    - Uses position encoding for distance matrix entries or sequence positions
    - Employs attention mechanism for sequence-length invariance
    - Produces fixed-size output regardless of input protein size
    """
    BLOCKS = ["norm_in", "encoder",]
    
    def __init__(
        self,
        input_dim: int,
        mlcv_dim: int = 2,
        d_model: int = 256,
        d_pair: int = 128, 
        n_head: int = 8,
        dim_feedforward: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        dim_normalization: bool = False,
        normalization_factor: float = 1.0,
        input_type: str = "distances",
        coordinate_processing: str = "to_distances",
        **kwargs,
    ):
        """
        Args:
            input_dim: Input dimension (ignored for distances, used for validation for coordinates)
            mlcv_dim: Output dimension of the collective variable
            d_model: Model dimension for attention layers
            d_pair: Pair representation dimension
            n_head: Number of attention heads
            dim_feedforward: Feedforward network dimension
            n_layers: Number of SA encoder layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for position encoding
            dim_normalization: Whether to apply dimension normalization
            normalization_factor: Factor for dimension normalization
            input_type: Type of input data - "distances" for pairwise distances or "coordinates" for 3D coordinates
            coordinate_processing: How to process coordinates - "to_distances" or "direct" (only used when input_type="coordinates")
        """
        super().__init__(in_features=input_dim, out_features=mlcv_dim)
        
        # Validate input_type and coordinate_processing
        if input_type not in ["distances", "coordinates"]:
            raise ValueError(f"input_type must be 'distances' or 'coordinates', got {input_type}")
        
        if coordinate_processing not in ["to_distances", "direct"]:
            raise ValueError(f"coordinate_processing must be 'to_distances' or 'direct', got {coordinate_processing}")
        
        self.input_dim = input_dim
        self.mlcv_dim = mlcv_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.input_type = input_type
        self.coordinate_processing = coordinate_processing
        
        # Input preprocessing layers
        self.distance_embedding = torch.nn.Linear(1, d_model)
        self.coordinate_embedding = torch.nn.Linear(3, d_model)  # For direct coordinate processing
        
        # Position encoding for distance matrix indices
        self.pos_encoding = PositionalEncodingMatrix(d_model, max_seq_len)
        
        # Initial projection to pair dimension 
        self.pair_projection = torch.nn.Linear(d_model, d_pair)
        
        # Structure-aware encoder layers
        self.encoder_layers = torch.nn.ModuleList([
            SAEncoderLayer(
                d_model=d_model,
                d_pair=d_pair, 
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Global pooling and output projection
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)  # Pool over sequence length
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_feedforward, mlcv_dim)
        )
        
        # Optional dimension normalization
        if dim_normalization:
            self.postprocessing = DIM_NORMALIZATION(
                feature_dim=mlcv_dim,
                normalization_factor=normalization_factor,
            )
    
    def coordinates_to_distances(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Convert 3D coordinates to pairwise distances.
        
        Args:
            coordinates: [B, N, 3] 3D coordinates
            
        Returns:
            distances: [B, n_pairs] pairwise distances where n_pairs = N * (N-1) / 2
        """
        batch_size, n_residues, coord_dim = coordinates.shape
        
        if coord_dim != 3:
            raise ValueError(f"Expected coordinates with shape [B, N, 3], got shape {coordinates.shape}")
        
        # Compute pairwise distances
        # coordinates: [B, N, 3]
        # Expand dimensions for broadcasting: [B, N, 1, 3] and [B, 1, N, 3]
        coord_i = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        coord_j = coordinates.unsqueeze(1)  # [B, 1, N, 3]
        
        # Compute squared distances: [B, N, N]
        dist_matrix = torch.norm(coord_i - coord_j, dim=-1)
        
        # Extract upper triangular part (excluding diagonal)
        distances = []
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distances.append(dist_matrix[:, i, j])
        
        # Stack to get [B, n_pairs]
        distances = torch.stack(distances, dim=1)
        
        return distances
    
    def coordinates_to_sequence_representation(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert 3D coordinates directly to sequence representation for SA layers.
        
        Args:
            coordinates: [B, N, 3] 3D coordinates
            
        Returns:
            x1d: [B, L, d_model] sequence representation
            x2d: [B, L, L, d_pair] pair representation  
            mask: [B, L] sequence mask
        """
        batch_size, n_residues, coord_dim = coordinates.shape
        
        if coord_dim != 3:
            raise ValueError(f"Expected coordinates with shape [B, N, 3], got shape {coordinates.shape}")
        
        # Embed coordinates directly: [B, N, 3] -> [B, N, d_model]
        coord_embedded = self.coordinate_embedding(coordinates)  # [B, N, d_model]
        
        # Add positional encoding for sequence positions
        # For coordinates, we use a simplified positional encoding based on sequence position
        pos_encoding_1d = torch.zeros(n_residues, self.d_model, device=coordinates.device, dtype=coordinates.dtype)
        for i in range(n_residues):
            for k in range(0, self.d_model, 2):
                if k < self.d_model:
                    pos_encoding_1d[i, k] = math.sin(i / (10000 ** (k / self.d_model)))
                if k + 1 < self.d_model:
                    pos_encoding_1d[i, k + 1] = math.cos(i / (10000 ** ((k + 1) / self.d_model)))
        
        # Create sequence representation
        x1d = coord_embedded + pos_encoding_1d.unsqueeze(0)  # [B, N, d_model]
        
        # Create pair representation from coordinate differences
        # Compute relative vectors between all pairs
        coord_i = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        coord_j = coordinates.unsqueeze(1)  # [B, 1, N, 3]
        relative_vectors = coord_i - coord_j  # [B, N, N, 3]
        
        # Embed relative vectors and project to pair dimension
        rel_embedded = self.coordinate_embedding(relative_vectors)  # [B, N, N, d_model]
        x2d = self.pair_projection(rel_embedded)  # [B, N, N, d_pair]
        
        # Create mask (all positions are valid for coordinates)
        mask = torch.ones(batch_size, n_residues, device=coordinates.device, dtype=torch.bool)
        
        return x1d, x2d, mask
    
    def cad_to_sequence_representation(self, cad_distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert pairwise CA distances to sequence representation for SA layers.
        
        Args:
            cad_distances: [B, n_pairs] where n_pairs = n_residues * (n_residues - 1) / 2
            
        Returns:
            x1d: [B, L, d_model] sequence representation
            x2d: [B, L, L, d_pair] pair representation  
            mask: [B, L] sequence mask
        """
        batch_size = cad_distances.shape[0]
        n_pairs = cad_distances.shape[1]
        
        # Reconstruct sequence length from number of pairs
        # n_pairs = n * (n-1) / 2, solve for n
        n_residues = int((1 + math.sqrt(1 + 8 * n_pairs)) / 2)
        
        # Embed distance values
        dist_embedded = self.distance_embedding(cad_distances.unsqueeze(-1))  # [B, n_pairs, d_model]
        
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
        pos_encoded = self.pos_encoding(n_residues)  # [L, L, d_model]
        distance_matrix = distance_matrix + pos_encoded.unsqueeze(0)
        
        # Create sequence representation by averaging over pairs
        x1d = distance_matrix.mean(dim=2)  # [B, L, d_model]
        
        # Create pair representation
        x2d = self.pair_projection(distance_matrix)  # [B, L, L, d_pair]
        
        # Create mask (all positions are valid for distance matrices)
        mask = torch.ones(batch_size, n_residues, device=cad_distances.device, dtype=torch.bool)
        
        return x1d, x2d, mask
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transferable MLCV.
        
        Args:
            input_data: Either [B, n_pairs] pairwise distances or [B, N, 3] 3D coordinates
            
        Returns:
            cv: [B, mlcv_dim] collective variables
        """
        # Process input based on type and processing method
        if self.input_type == "coordinates":
            # Input is 3D coordinates [B, N, 3]
            if len(input_data.shape) != 3 or input_data.shape[-1] != 3:
                raise ValueError(f"For input_type='coordinates', expected shape [B, N, 3], got {input_data.shape}")
            
            if self.coordinate_processing == "direct":
                # Process coordinates directly without converting to distances
                x1d, x2d, mask = self.coordinates_to_sequence_representation(input_data)
            elif self.coordinate_processing == "to_distances":
                # Convert to distances first, then process
                cad_distances = self.coordinates_to_distances(input_data)
                x1d, x2d, mask = self.cad_to_sequence_representation(cad_distances)
            else:
                raise ValueError(f"Unknown coordinate_processing: {self.coordinate_processing}")
                
        elif self.input_type == "distances":
            # Input is already distances [B, n_pairs]
            if len(input_data.shape) != 2:
                raise ValueError(f"For input_type='distances', expected shape [B, n_pairs], got {input_data.shape}")
            x1d, x2d, mask = self.cad_to_sequence_representation(input_data)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")
        
        # Create dummy pose (identity rotations and zero translations)
        batch_size, seq_len = x1d.shape[:2]
        device = x1d.device
        
        dummy_translations = torch.zeros(batch_size, seq_len, 3, device=device)
        dummy_rotations = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        pose = (dummy_translations, dummy_rotations)
        
        # Create attention bias (no masking for distance matrices)
        bias = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)  # [B, 1, L, L]
        
        # Apply SA encoder layers
        for layer in self.encoder_layers:
            x1d = layer(x1d, x2d, pose, bias)
        
        # Global pooling over sequence length
        # x1d: [B, L, d_model] -> [B, d_model, L] -> [B, d_model, 1] -> [B, d_model]
        pooled = self.global_pool(x1d.transpose(1, 2)).squeeze(-1)
        
        # Output projection
        cv = self.output_projection(pooled)  # [B, mlcv_dim]
        
        # Apply post-processing if available
        if hasattr(self, 'postprocessing') and self.postprocessing is not None:
            cv = self.postprocessing(cv)
            
        return cv


class PositionalEncodingMatrix(torch.nn.Module):
    """
    Positional encoding for pairwise distance matrices.
    
    Encodes the relative positions (i, j) in the distance matrix 
    to help the model understand spatial relationships.
    """
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix [max_len, max_len, d_model]
        pe = torch.zeros(max_len, max_len, d_model)
        
        for i in range(max_len):
            for j in range(max_len):
                # Encode both absolute positions and relative distance
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
        """
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encoding matrix [seq_len, seq_len, d_model]
        """
        return self.pe[:seq_len, :seq_len, :]


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
    parser.add_argument('--mlcv_dim', type=int, default=1, help='MLCV dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--transferable', type=bool, default=False, help='Transferable model')
    
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
    elif args.molecule == "2F4K":
        INPUT_DIM = 595
    else:
        raise ValueError(f"Molecule {args.molecule} not supported")
    MLCV_DIM = args.mlcv_dim
    
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
    print("State dict keys:", list(mlcv_state_dict.keys()))
    
    # Check if postprocessing exists in state dict
    has_postprocessing = any(key.startswith('postprocessing') for key in mlcv_state_dict.keys())
    
    if has_postprocessing:
        print("Found postprocessing in state dict, adding postprocessing module to model...")
        # Add a dummy postprocessing module before loading state dict
        mlcv_dim = encoder_layers[-1]
        model.postprocessing = PostProcess(feature_dim=mlcv_dim)
    
    # Load state dict with strict=False to handle missing keys gracefully
    missing_keys, unexpected_keys = model.load_state_dict(mlcv_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    model = model.to(device)
    model.eval()
    
    # Load molecule data
    projection_data = load_molecule_data(args.molecule, device)
    
    # Only add postprocessing if it doesn't already exist
    if not has_postprocessing:
        print("No postprocessing found in model, adding new postprocessing...")
        model = add_postprocessing_to_model(model, projection_data, device)
    else:
        print("Model already has postprocessing from loaded state dict.")
    
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
