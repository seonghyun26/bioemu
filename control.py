import os
import wandb
import hydra
import torch
import yaml
import torch.nn as nn
import mdtraj as md

from datetime import datetime
from pathlib import Path
from torch_geometric.data import Batch
from tqdm import tqdm
from omegaconf import OmegaConf,open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from bioemu.chemgraph import ChemGraph
from bioemu.sample import get_context_chemgraph
from bioemu.denoiser import get_score, dpm_solver
from bioemu.sde_lib import SDE
from bioemu.chemgraph import ChemGraph
from bioemu.models import DiGConditionalScoreModel
from bioemu.so3_sde import SO3SDE


from data import *
from model import *

# For post-processing
from mlcolvar.core.transform import Statistics, Transform


def sanitize_range(range_tensor: torch.Tensor) -> torch.Tensor:
    """Sanitize range tensor to avoid division by zero"""
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


# SINGLE_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_single.npy"
# PAIR_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_pair.npy"
# rollout_config_path = "/home/shpark/prj-mlcv/lib/bioemu/notebook/rollout.yaml"
# OUTPUT_DIR = Path("~/prj-mlcv/lib/bioemu/ppft_example_output").expanduser()
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# cln025_alpha_carbon_idx = [4, 25, 46, 60, 72, 87, 101, 108, 122, 146]
    
    
def kabsch_rmsd(
    P: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)
    
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))  # B
    # Vt[d < 0.0, -1] *= -1.0
    if d < 0:
        Vt = Vt.clone()
        Vt[-1] = Vt[-1] * -1

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))

    # Calculate RMSD
    P_aligned = torch.matmul(P, R.transpose(-2, -1)) + t
    rmsd = (P_aligned - Q).square().sum(-1).mean(-1).sqrt()
    
    return rmsd


def coord2rot(
    pdb,
    coordinates: torch.Tensor,
):
    """
    Computes a rotation matrix Q from C_alpha, N, and C atom coordinates
    using the Gram-Schmidt algorithm.

    Args:
        a (torch.Tensor): Tensor of C_alpha atom coordinates with shape (10, 3).
        b (torch.Tensor): Tensor of N atom coordinates with shape (10, 3).
        c (torch.Tensor): Tensor of C atom coordinates with shape (10, 3).

    Returns:
        torch.Tensor: A rotation matrix Q with shape (10, 3, 3) where each 
                      (3, 3) matrix is in SO(3).
    """
    ca_indices = pdb.topology.select('name CA')
    n_indices = pdb.topology.select('name N')
    c_indices = pdb.topology.select('name C')
    pdb_xyz = torch.tensor(pdb.xyz[0])

    a = pdb_xyz[ca_indices]
    b = pdb_xyz[n_indices]
    c = pdb_xyz[c_indices]

    # Get displacement vectors
    u = b - a  # C_alpha -> N
    v = c - a  # C_alpha -> C

    # Gram-Schmidt Process
    # First basis vector
    e1 = u / torch.norm(u, dim=-1, keepdim=True)

    # Second basis vector
    u2 = v - torch.sum(e1 * v, dim=-1, keepdim=True) * e1
    e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)

    # Third basis vector (cross product of e1 and e2)
    e3 = torch.cross(e1, e2, dim=-1)

    # Construct the rotation matrix Q
    # Stack the orthonormal basis vectors to form the columns of Q
    Q = torch.stack([e1, e2, e3], dim=-1)

    # To ensure the determinant is +1 (SO(3)), we need to check the orientation.
    # The Gram-Schmidt process doesn't guarantee a right-handed system, 
    # but the image description implies one. The cross-product implicitly
    # handles this by creating a vector orthogonal to the plane of e1 and e2.
    # The resulting matrix will have a determinant of +1 or -1. 
    # For a right-handed basis, the determinant is +1.
    
    # We can explicitly check and correct if needed, but the cross product 
    # as used here for the third vector gives a right-handed system by definition.
    # Therefore, the determinant will be +1, and the resulting matrix Q
    # is a valid rotation matrix in SO(3).

    return Q

def calc_standard_denoising_loss(
    score_model: torch.nn.Module,
    mlcv: torch.Tensor,
    target: torch.Tensor,
    orientation: torch.Tensor,
    sdes: dict[str, SDE],
    batch: list[ChemGraph],
    cfg: OmegaConf = None,
    condition_mode: str = "none",
    min_t: float = 1e-4,
    max_t: float = 0.99,
) -> tuple[torch.Tensor, dict]:
    """
    Classic denoising score-matching loss for fine-tuning with time-lagged data.
    
    This implements the standard denoising diffusion training objective for both
    positions and orientations:
    L = E_t,x0,ε [ ||s_θ(x_t, t) - ∇_{x_t} log p(x_t | x_0)||² ]
    
    For positions (Gaussian forward process):
    - x_t = α_t x_0 + σ_t ε, where ε ~ N(0, I)
    - ∇_{x_t} log p(x_t | x_0) = -(x_t - α_t x_0) / σ_t²
    
    For orientations (SO3 diffusion):
    - R_t ~ IGSO3(R_0, σ_t), where σ_t is the marginal std at time t
    - ∇_{R_t} log p(R_t | R_0) computed using SO3SDE.compute_score()
    
    Args:
        score_model: The score network to fine-tune
        mlcv: MLCV conditioning information (if using conditional model)
        target: Time-lagged target positions (x_0 in the diffusion notation)
        sdes: Dictionary of SDEs for the forward process
        batch: List of ChemGraph templates
        cfg: Configuration
        condition_mode: How to condition on MLCV
        min_t: Minimum timestep to sample
        max_t: Maximum timestep to sample
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    device = batch[0].pos.device
    batch_size = len(batch)
    target = target.to(device)
    clean_graphs: list[ChemGraph] = []
    
    if cfg.data.representation == "cad-pos":
        # Target contains all atom positions, we need CA only
        pdb = md.load_pdb(f"/home/shpark/prj-mlcv/lib/DESRES/data/{cfg.data.system_id}/{cfg.data.system_id}_from_mae.pdb")
        ca_indices = pdb.topology.select('name CA')
        
        for i, template_graph in enumerate(batch):
            clean_pos = target[i, ca_indices].to(device)            
            # DEBUGGING: Set orientations to identity matrix to isolate position learning
            clean_orientations = orientation[i].to(device)
            # clean_orientations = torch.eye(3, device=device).unsqueeze(0).repeat(clean_pos.shape[0], 1, 1)
            clean_graphs.append(template_graph.replace(
                pos=clean_pos,
                node_orientations=clean_orientations,
            ))
    else:
        raise ValueError(f"Representation {cfg.data.representation} not supported")
    
    clean_batch = Batch.from_data_list(clean_graphs)
    # t = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
    t = torch.randint(1, 100, (batch_size,), device=device) / 100.0
    alpha_t, sigma_t = sdes["pos"].mean_coeff_and_std(
        x=clean_batch.pos,
        t=t,
        batch_idx=clean_batch.batch,
    )
    
    # Sample Gaussian noise for positions
    noise = torch.randn_like(clean_batch.pos)
    x_t = alpha_t * clean_batch.pos + sigma_t * noise
    noisy_batch = clean_batch.replace(
        pos=x_t,
    )
    
    # Predict score using the model
    predicted = get_score(
        batch=noisy_batch,
        sdes=sdes,
        score_model=score_model,
        t=t,
        mlcv=mlcv,
    )
    
    # Compute target scores for positions
    pos_target_score = -(x_t - alpha_t * clean_batch.pos) / (sigma_t ** 2)
    pos_loss = torch.nn.functional.mse_loss(predicted["pos"], pos_target_score)    
    total_loss = pos_loss
    
    metrics = {
        "pos_loss": pos_loss.item(),
        "timestep_mean": t.mean().item(),
        "timestep_std": t.std().item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, metrics


# NOTE: Fine-tuning in ppft matter
def calc_tlc_loss(
    score_model: torch.nn.Module,
    mlcv: torch.Tensor,
    target: torch.Tensor,
    sdes: dict[str, SDE],
    batch: list[ChemGraph],
    n_replications: int,
    mid_t: float,
    N_rollout: int,
    record_grad_steps: set[int],
    cfg: OmegaConf = None,
    condition_mode: str = "none",
) -> tuple[torch.Tensor, dict]:
    device = batch[0].pos.device
    assert isinstance(batch, list)  # Not a Batch!

    # Debug: Check MLCV at entry to calc_tlc_loss
    if hasattr(cfg, 'log') and cfg.log.debug:
        print(f"  [calc_tlc_loss] MLCV shape: {mlcv.shape if mlcv is not None else None}")
        print(f"  [calc_tlc_loss] MLCV range: [{mlcv.min().item():.6f}, {mlcv.max().item():.6f}]" if mlcv is not None else "  [calc_tlc_loss] MLCV is None")
        print(f"  [calc_tlc_loss] condition_mode: {condition_mode}")

    x_in = Batch.from_data_list(batch * n_replications)
    
    # CRITICAL FIX: Replicate MLCV to match batch replication
    if mlcv is not None and n_replications > 1:
        mlcv_replicated = mlcv.repeat(n_replications, 1)  # Repeat MLCV for each replication
    else:
        mlcv_replicated = mlcv
    
    # Debug: Check batch replication and MLCV alignment
    if hasattr(cfg, 'log') and cfg.log.debug:
        print(f"  [calc_tlc_loss] Original batch size: {len(batch)}, n_replications: {n_replications}")
        print(f"  [calc_tlc_loss] x_in.num_graphs: {x_in.num_graphs}")
        print(f"  [calc_tlc_loss] MLCV batch size: {mlcv.shape[0] if mlcv is not None else None}")
        print(f"  [calc_tlc_loss] MLCV_replicated batch size: {mlcv_replicated.shape[0] if mlcv_replicated is not None else None}")
        print(f"  [calc_tlc_loss] Total samples generated: {len(batch)} × {n_replications} = {x_in.num_graphs}")
        if mlcv_replicated is not None and mlcv_replicated.shape[0] != x_in.num_graphs:
            print(f"  ⚠️  WARNING: MLCV batch size mismatch! MLCV: {mlcv_replicated.shape[0]}, x_in: {x_in.num_graphs}")
    
    x0 = _rollout(
        batch=x_in,
        sdes=sdes,
        score_model=score_model,
        mid_t=mid_t,
        N_rollout=N_rollout,
        device=device,
        mlcv=mlcv_replicated,  # Use replicated MLCV
        condition_mode=condition_mode,
        record_grad_steps=record_grad_steps,
        cfg=cfg,
    )
    num_systems_sampled = len(batch)
    
    # =================================================================
    # STRUCTURE QUALITY MONITORING: Compute CA and C-N distances
    # =================================================================
    structure_metrics = {}
    
    try:
        # Load reference structure to get atom indices
        pdb = md.load_pdb(f"/home/shpark/prj-mlcv/lib/DESRES/data/{cfg.data.system_id}/{cfg.data.system_id}_from_mae.pdb")
        ca_indices = pdb.topology.select('name CA')
        c_indices = pdb.topology.select('name C')
        n_indices = pdb.topology.select('name N')
        
        # Collect metrics across all generated samples
        all_ca_sequential_distances = []
        all_cn_distances = []
        all_ca_pairwise_distances = []
        
        for idx in range(num_systems_sampled):
            for rep in range(n_replications):
                sample_idx = idx + rep * num_systems_sampled
                generated_pos = x0.get_example(sample_idx).pos  # All atom positions [N_atoms, 3]
                
                # Extract CA positions
                generated_ca_pos = generated_pos  # For CLN025, these should be CA positions already
                
                # 1. Sequential CA-CA distances
                ca_seq_dists = torch.norm(generated_ca_pos[1:] - generated_ca_pos[:-1], dim=1)
                all_ca_sequential_distances.extend(ca_seq_dists.detach().cpu().numpy())
                
                # 2. All pairwise CA distances  
                ca_pairwise_dists = torch.cdist(generated_ca_pos, generated_ca_pos, p=2)
                # Get upper triangular part (excluding diagonal)
                n_ca = ca_pairwise_dists.shape[0]
                i, j = torch.triu_indices(n_ca, n_ca, offset=1)
                all_ca_pairwise_distances.extend(ca_pairwise_dists[i, j].detach().cpu().numpy())
                
                # 3. C-N peptide bond distances (if we have full backbone)
                if len(generated_pos) > len(ca_indices):  # Full backbone available
                    try:
                        generated_c_pos = generated_pos[c_indices]
                        generated_n_pos = generated_pos[n_indices]
                        # C(i) to N(i+1) distances
                        cn_dists = torch.norm(generated_c_pos[:-1] - generated_n_pos[1:], dim=1)
                        all_cn_distances.extend(cn_dists.detach().cpu().numpy())
                    except:
                        pass  # Skip if indexing fails
        
        # Compute statistics
        if all_ca_sequential_distances:
            ca_seq_dists = torch.tensor(all_ca_sequential_distances)
            structure_metrics.update({
                'ca_sequential_dist_mean': ca_seq_dists.mean().item(),
                'ca_sequential_dist_std': ca_seq_dists.std().item(),
                'ca_sequential_dist_min': ca_seq_dists.min().item(),
                'ca_sequential_dist_max': ca_seq_dists.max().item(),
                'ca_sequential_dist_violations': (ca_seq_dists > 0.5).sum().item() / len(ca_seq_dists),  # >5Å violations
            })
        
        if all_ca_pairwise_distances:
            ca_pair_dists = torch.tensor(all_ca_pairwise_distances)
            structure_metrics.update({
                'ca_pairwise_dist_mean': ca_pair_dists.mean().item(),
                'ca_pairwise_dist_std': ca_pair_dists.std().item(),
                'ca_pairwise_dist_max': ca_pair_dists.max().item(),
            })
        
        if all_cn_distances:
            cn_dists = torch.tensor(all_cn_distances)
            structure_metrics.update({
                'cn_bond_dist_mean': cn_dists.mean().item(),
                'cn_bond_dist_std': cn_dists.std().item(),
                'cn_bond_dist_min': cn_dists.min().item(),
                'cn_bond_dist_max': cn_dists.max().item(),
                'cn_bond_violations': (cn_dists > 0.18).sum().item() / len(cn_dists),  # >1.8Å violations
            })
        
        # Debug output
        if cfg.log.debug and structure_metrics:
            print(f"=== STRUCTURE QUALITY METRICS ===")
            for key, value in structure_metrics.items():
                print(f"{key}: {value:.4f}")
            print(f"=== END STRUCTURE METRICS ===")
            
    except Exception as e:
        print(f"Warning: Could not compute structure metrics: {e}")
        structure_metrics = {}
    
    # =================================================================
    
    if cfg.data.representation == "cad":
        generated_ca_distance = torch.empty(num_systems_sampled, n_replications, cfg.data.input_dim, device=device)
        for idx in range(num_systems_sampled):
            for rep in range(n_replications):
                sample_idx = idx + rep * num_systems_sampled
                generated_ca_pos = x0.get_example(sample_idx).pos
                generated_ca_pair_distances = torch.cdist(generated_ca_pos, generated_ca_pos, p=2)
                n = generated_ca_pair_distances.shape[0]
                i, j = torch.triu_indices(n, n, offset=1)
                generated_ca_distance[idx, rep] = generated_ca_pair_distances[i, j]
        generated_ca_distance_mean = generated_ca_distance.mean(dim=1)
        loss = torch.nn.functional.mse_loss(generated_ca_distance_mean, target)
    
    elif cfg.data.representation == "cad-pos":
        pdb = md.load_pdb(f"/home/shpark/prj-mlcv/lib/DESRES/data/{cfg.data.system_id}/{cfg.data.system_id}_from_mae.pdb")
        ca_indices = pdb.topology.select('name CA')
        
        loss = torch.tensor(0.0, device=device)
        for idx in range(num_systems_sampled):
            system_loss = torch.tensor(0.0, device=device)
            target_ca_pos = target[idx, ca_indices]
            for rep in range(n_replications):
                sample_idx = idx + rep * num_systems_sampled
                generated_ca_pos = x0.get_example(sample_idx).pos
                target_ca_pos = target[idx, ca_indices]
                system_loss = system_loss + kabsch_rmsd(generated_ca_pos, target_ca_pos)
            loss = loss + (system_loss / n_replications)
        loss = loss / num_systems_sampled
    
    else:
        raise ValueError(f"Representation {cfg.data.representation} not supported")
    
    # seq_idx = torch.arange(n-1)
    # generated_ca_seq_distances = generated_ca_pair_distances[seq_idx, seq_idx+1]
    # generated_ca_distances_distribution = generated_ca_distances_distribution + generated_ca_seq_distances
    # generated_ca_distances_sum = generated_ca_distances_sum + generated_ca_seq_distances.mean()
        
    # loss = loss / num_systems_sampled
    # generated_ca_distances_sum = generated_ca_distances_sum / num_systems_sampled
    # generated_ca_distances_distribution = generated_ca_distances_distribution / num_systems_sampled
    
    return loss, structure_metrics


def _rollout(
    batch: Batch,
    sdes: dict[str, SDE],
    score_model,
    mid_t: float,
    N_rollout: int,
    device: torch.device,
    mlcv: torch.Tensor,
    condition_mode: str = "none",
    record_grad_steps: set[int] = set(),
    cfg: OmegaConf = None,
):
    """Fast rollout to get a sampled structure in a small number of steps.
    Note that in the last step, only the positions are calculated, and not the orientations,
    because the orientations are not used to compute foldedness.
    """
    batch_size = batch.num_graphs
    
    # Debug: Check MLCV at entry to _rollout
    if mlcv is not None and hasattr(score_model, 'training') and score_model.training:
        print(f"    [_rollout] MLCV passed to dpm_solver: shape={mlcv.shape}, condition_mode={condition_mode}")
        print(f"    [_rollout] MLCV values: {mlcv.flatten()[:5].detach().cpu().numpy()}")
    
    # Perform a few denoising steps to get a partially denoised sample `x_mid`.
    x_mid: ChemGraph = dpm_solver(
        sdes=sdes,
        batch=batch,
        eps_t=mid_t,
        max_t=0.99,
        N=N_rollout,
        device=device,
        score_model=score_model,
        mlcv=mlcv,
        condition_mode=condition_mode,
        record_grad_steps=record_grad_steps,
        cfg=cfg,
    )

    # Predict clean x (x0) from x_mid in a single jump.
    # This step is always with gradient.
    mid_t_expanded = torch.full((batch_size,), mid_t, device=device)
    score_mid_t = get_score(
        batch=x_mid,
        sdes=sdes,
        t=mid_t_expanded,
        score_model=score_model,
        mlcv=mlcv,
    )["pos"]

    # No need to compute orientations, because they are not used to compute foldedness.
    x0_pos = _get_x0_given_xt_and_score(
        sde=sdes["pos"],
        x=x_mid.pos,
        t=torch.full((batch_size,), mid_t, device=device),
        batch_idx=x_mid.batch,
        score=score_mid_t,
    )

    return x_mid.replace(pos=x0_pos)


def _get_x0_given_xt_and_score(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x_0 given x_t and score.
    """
    assert not isinstance(sde, SO3SDE)

    alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx)

    return (x + sigma_t**2 * score) / alpha_t
    

def _add_zero_conv_mlp(
    init_mode: str,
    hidden_dim: int,
    mlcv_dim: int,
):
    """
    Create a conditioning MLP with specified initialization.
    
    Args:
        init_mode: Initialization method ('zero', 'rand', 'xavier', 'xavier_normal')
        hidden_dim: Hidden dimension of the model
        mlcv_dim: MLCV dimension
    """
    if init_mode == "zero":
        zero_linear = nn.Linear(hidden_dim + mlcv_dim, hidden_dim)
        nn.init.zeros_(zero_linear.weight)  # Zero initialization
        nn.init.zeros_(zero_linear.bias)  # Zero bias 
        zero_mlp = nn.Sequential(zero_linear)
    elif init_mode == "rand":
        zero_linear = nn.Linear(hidden_dim + mlcv_dim, hidden_dim)
        nn.init.normal_(zero_linear.weight, mean=0.0, std=1e-4)  # Very small random weights
        nn.init.zeros_(zero_linear.bias)  # Zero bias
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    elif init_mode == "xavier":
        zero_linear = nn.Linear(hidden_dim + mlcv_dim, hidden_dim)
        nn.init.xavier_uniform_(zero_linear.weight)  # Xavier uniform initialization
        nn.init.zeros_(zero_linear.bias)  # Zero bias
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    elif init_mode == "xavier_normal":
        zero_linear = nn.Linear(hidden_dim + mlcv_dim, hidden_dim)
        nn.init.xavier_normal_(zero_linear.weight)  # Xavier normal initialization
        nn.init.zeros_(zero_linear.bias)  # Zero bias
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    else:
        raise ValueError(f"Invalid initialization method: {init_mode}. Choose from ['zero', 'rand', 'xavier', 'xavier_normal']")
    
    return zero_mlp


def _add_orientation_mlp(
    init_mode: str,
    mlcv_dim: int,
):
    """
    Create orientation conditioning MLP (2 rotation features + mlcv_dim → 3 axis-angle).
    """
    if init_mode == "zero":
        zero_linear = nn.Linear(2 + mlcv_dim, 3)
        nn.init.zeros_(zero_linear.weight)
        nn.init.zeros_(zero_linear.bias)
        zero_mlp = nn.Sequential(zero_linear)
    elif init_mode == "rand":
        zero_linear = nn.Linear(2 + mlcv_dim, 3)
        nn.init.normal_(zero_linear.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(zero_linear.bias)
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    elif init_mode == "xavier":
        zero_linear = nn.Linear(2 + mlcv_dim, 3)
        nn.init.xavier_uniform_(zero_linear.weight)
        nn.init.zeros_(zero_linear.bias)
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    elif init_mode == "xavier_normal":
        zero_linear = nn.Linear(2 + mlcv_dim, 3)
        nn.init.xavier_normal_(zero_linear.weight)
        nn.init.zeros_(zero_linear.bias)
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    else:
        raise ValueError(f"Invalid initialization method: {init_mode}. Choose from ['zero', 'rand', 'xavier', 'xavier_normal']")
    
    return zero_mlp


def clip_loss(
    loss: torch.Tensor, 
    max_loss_value: float = 100.0, 
    min_loss_value: float = 0.0,
    enable_clipping: bool = True,
    debug: bool = False
) -> tuple[torch.Tensor, bool]:
    """
    Clip loss values to prevent training instability from extreme losses.
    
    Args:
        loss: The loss tensor to clip
        max_loss_value: Maximum allowed loss value
        min_loss_value: Minimum allowed loss value (should be >= 0 for most losses)
        enable_clipping: Whether to actually perform clipping
        debug: Whether to print debug information
    
    Returns:
        Tuple of (clipped_loss, was_clipped)
    """
    if not enable_clipping:
        return loss, False
    
    original_loss = loss.item()
    was_clipped = False
    
    if loss > max_loss_value:
        loss = torch.tensor(max_loss_value, device=loss.device, dtype=loss.dtype, requires_grad=True)
        was_clipped = True
        if debug:
            print(f"⚠️  Loss clipped from {original_loss:.6f} to {max_loss_value:.6f} (max clipping)")
    elif loss < min_loss_value:
        loss = torch.tensor(min_loss_value, device=loss.device, dtype=loss.dtype, requires_grad=True)
        was_clipped = True
        if debug:
            print(f"⚠️  Loss clipped from {original_loss:.6f} to {min_loss_value:.6f} (min clipping)")
    
    return loss, was_clipped


def add_postprocessing_module(mlcv_model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: torch.device, cfg: OmegaConf = None, reference_frame_path: str = None):
    """
    Add post-processing module to MLCV model based on full dataset statistics.
    
    Args:
        mlcv_model: The trained MLCV model
        dataset: The dataset to compute statistics from
        device: Device to run computations on
        cfg: Configuration object
        reference_frame_path: Optional path to reference frame for sign flipping
    """
    print("Computing post-processing statistics on full dataset...")
    
    # Load full dataset for statistics computation
    if cfg and hasattr(cfg.data, 'system_id'):
        molecule = cfg.data.system_id
        projection_data_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-cad.pt"
        
        if os.path.exists(projection_data_path):
            projection_data = torch.load(projection_data_path).to(device)
            print(f"Loaded full dataset from {projection_data_path}: {projection_data.shape}")
        else:
            print(f"Warning: Full dataset not found at {projection_data_path}")
            print("Using current dataset for statistics...")
            # Use current dataset as fallback
            all_data = []
            for batch in dataset:
                all_data.append(batch["current_data"])
            projection_data = torch.cat(all_data, dim=0).to(device)
    else:
        print("Using current dataset for statistics...")
        # Use current dataset as fallback
        all_data = []
        for batch in dataset:
            all_data.append(batch["current_data"])
        projection_data = torch.cat(all_data, dim=0).to(device)
    
    # Evaluate model on full dataset using batch processing
    mlcv_model.eval()
    batch_size = 10000
    print(f"Computing CV values for post-processing in batches of {batch_size}...")
    dataset = TensorDataset(projection_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get output dimension from a sample
    with torch.no_grad():
        sample_batch = next(iter(dataloader))[0]
        sample_output = mlcv_model(sample_batch)
        output_dim = sample_output.shape[1]
    cv_batches = torch.zeros((len(projection_data), output_dim)).to(projection_data.device)
    
    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(tqdm(
            dataloader,
            desc="Computing CV values for post-processing",
            total=len(dataloader),
            leave=False,
        )):
            batch_cv = mlcv_model(batch_data)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_cv.shape[0]  # Handle last batch size correctly
            cv_batches[start_idx:end_idx] = batch_cv
    
    cv = cv_batches
    print(f"CV shape: {cv.shape}")
    print(f"CV range: [{cv.min():.6f}, {cv.max():.6f}]")
    
    # Data for post-processing
    stats = Statistics(cv.cpu()).to_dict()
    reference_frame_cv = None
    if reference_frame_path and os.path.exists(reference_frame_path):
        try:
            ref_traj = md.load(reference_frame_path)
            ref_pos = ref_traj.xyz[0]  # Get first (and only) frame
            
            # Convert to the same representation as training data
            if cfg.data.representation == "cad":
                # Compute CA distances
                ca_indices = ref_traj.topology.select('name CA')
                ref_ca_pos = ref_pos[ca_indices]
                ref_distances = torch.cdist(torch.from_numpy(ref_ca_pos), torch.from_numpy(ref_ca_pos), p=2)
                n = ref_distances.shape[0]
                i, j = torch.triu_indices(n, n, offset=1)
                ref_cad = ref_distances[i, j].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    reference_frame_cv = mlcv_model(ref_cad).item()
                print(f"Reference frame CV value: {reference_frame_cv}")
        except Exception as e:
            print(f"Warning: Could not load reference frame from {reference_frame_path}: {e}")
            reference_frame_cv = None
    
    # Create and attach post-processing module
    mlcv_dim = cv.shape[1]
    postprocessing = PostProcess(
        stats=stats,
        reference_frame_cv=reference_frame_cv,
        feature_dim=mlcv_dim
    ).to(device)
    
    # Attach to model
    mlcv_model.postprocessing = postprocessing
    
    # Test post-processed output using batch processing
    print("Testing post-processed output...")
    postprocessed_cv_batches = torch.zeros((len(projection_data), output_dim)).to(projection_data.device)
    
    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(tqdm(
            dataloader,
            desc="Testing post-processed CV",
            total=len(dataloader),
            leave=False,
        )):
            batch_postprocessed_cv = mlcv_model(batch_data)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_postprocessed_cv.shape[0]
            postprocessed_cv_batches[start_idx:end_idx] = batch_postprocessed_cv
    
    postprocessed_cv = postprocessed_cv_batches
    print(f"Post-processed CV range: [{postprocessed_cv.min():.6f}, {postprocessed_cv.max():.6f}]")
    
    # Log post-processing statistics to wandb
    postprocessing_stats = {
        "postprocessing/original_cv_min": cv.min().item(),
        "postprocessing/original_cv_max": cv.max().item(),
        "postprocessing/original_cv_mean": cv.mean().item(),
        "postprocessing/original_cv_std": cv.std().item(),
        "postprocessing/processed_cv_min": postprocessed_cv.min().item(),
        "postprocessing/processed_cv_max": postprocessed_cv.max().item(),
        "postprocessing/processed_cv_mean": postprocessed_cv.mean().item(),
        "postprocessing/processed_cv_std": postprocessed_cv.std().item(),
        "postprocessing/normalization_mean": stats["mean"].item() if stats["mean"].numel() == 1 else stats["mean"][0].item(),
        "postprocessing/normalization_range": postprocessing.range.item() if postprocessing.range.numel() == 1 else postprocessing.range[0].item(),
        "postprocessing/flip_sign": postprocessing.flip_sign.item(),
    }
    
    if reference_frame_cv is not None:
        postprocessing_stats["postprocessing/reference_cv_value"] = reference_frame_cv
    
    wandb.log(postprocessing_stats)
    
    print("Post-processing module attached successfully!")
    
    return mlcv_model


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.model.training.seed
    torch.manual_seed(seed)
    
    # Set logging
    date = cfg.log.date
    if date == "debug":
        pass
    elif date == "now":
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.log.date = datetime.now().strftime("%m%d_%H%M%S")
        os.makedirs(f"model/{cfg.log.date}")
    else:
        os.makedirs(f"model/{cfg.log.date}")
    
    run = wandb.init(
        project="bioemu-ctrl",
        entity="eddy26",
        tags=cfg.log.tags,
        config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )
    
    # score model
    ckpt_path = cfg.model.score_model.ckpt_path
    cfg_path = cfg.model.score_model.cfg_path
    with open(cfg_path) as f:
        model_config = yaml.safe_load(f)
    model_config["score_model"]["condition_mode"] = cfg.model.mlcv_model.condition_mode
    wandb.config.update(model_config)
    model_state = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    score_model = score_model.to(device)  # CRITICAL FIX: Move score model to device
    
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])
    for sde_name, sde in sdes.items():
        if hasattr(sde, 'to'):
            sde.to(device)
        # For SO3SDE, ensure internal components are on device
        if hasattr(sde, 'score_function') and hasattr(sde.score_function, 'to'):
            sde.score_function.to(device)
        if hasattr(sde, 'igso3') and hasattr(sde.igso3, 'to'):
            sde.igso3.to(device)
    
    if cfg.model.score_model.mode == "train":
        score_model.train()
    elif cfg.model.score_model.mode == "eval":
        score_model.eval()
    else:
        raise ValueError(f"Score model mode {cfg.model.score_model.mode} not supported")
    # if cfg.log.debug:
        # score_model.model_nn._debug_conditioning = True
        # score_model.model_nn.st_module._debug_conditioning = True
        
    
    # add zero conv MLP to score model
    if cfg.model.mlcv_model.condition_mode in ["backbone", "backbone-both"]:
        hidden_dim = 3
    elif cfg.model.mlcv_model.condition_mode in ["input", "latent", "input-control"]:
        hidden_dim = 512
    else:
        raise ValueError(f"Condition type {cfg.model.mlcv_model.condition_mode} not supported")
    if cfg.model.mlcv_model.condition_mode in ["input", "latent", "backbone"]:
        zero_mlp = _add_zero_conv_mlp(
            init_mode=cfg.model.score_model.init,
            hidden_dim=hidden_dim,
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
        )
        score_model.model_nn.add_module(f"zero_conv_mlp", zero_mlp)
        zero_conv_mlp = score_model.model_nn.get_submodule("zero_conv_mlp")
        zero_conv_mlp.train()
        zero_conv_mlp.to(device)
        
    elif cfg.model.mlcv_model.condition_mode == "input-control":
        for i in range(8):
            zero_mlp = _add_zero_conv_mlp(
                init_mode=cfg.model.score_model.init,
                hidden_dim=hidden_dim,
                mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
            )
            zero_mlp.train()
            score_model.model_nn.st_module.encoder.add_module(f"zero_conv_mlp_{i}", zero_mlp)
            zero_mlp.to(device)
            
    elif cfg.model.mlcv_model.condition_mode == "backbone-both":
        zero_mlp_pos = _add_zero_conv_mlp(
            init_mode=cfg.model.score_model.init,
            hidden_dim=hidden_dim,
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
        )
        zero_mlp_pos.train()
        score_model.model_nn.add_module(f"zero_conv_mlp_pos", zero_mlp_pos)
        zero_conv_mlp_pos = score_model.model_nn.get_submodule("zero_conv_mlp_pos")
        zero_conv_mlp_pos.to(device)
        
        zero_mlp_orient = _add_orientation_mlp(
            init_mode=cfg.model.score_model.init,
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
        )
        zero_mlp_orient.train()
        score_model.model_nn.add_module(f"zero_conv_mlp_orient", zero_mlp_orient)
        zero_conv_mlp_orient = score_model.model_nn.get_submodule("zero_conv_mlp_orient")
        zero_conv_mlp_orient.to(device)
        
    for p in score_model.parameters():
        p.requires_grad = True
    
    if cfg.log.score_model.watch:
        if cfg.model.mlcv_model.condition_mode == "backbone-both":
            wandb.watch(
                zero_conv_mlp_pos,
                log=cfg.log.score_model.log,
                log_freq=cfg.log.score_model.watch_freq,
            )
            wandb.watch(
                zero_conv_mlp_orient,
                log=cfg.log.score_model.log,
                log_freq=cfg.log.score_model.watch_freq,
            )
        elif cfg.model.mlcv_model.condition_mode == "input-control":
            for i in range(8):
                wandb.watch(
                    score_model.model_nn.st_module.encoder.get_submodule(f"zero_conv_mlp_{i}"),
                    log=cfg.log.score_model.log,
                    log_freq=cfg.log.score_model.watch_freq,
                )
        else:
            wandb.watch(
                zero_conv_mlp,
                log=cfg.log.score_model.log,
                log_freq=cfg.log.score_model.watch_freq,
            )
    
    
    # MLCV model
    method = cfg.model.mlcv_model.name
    if method == "ours":
        mlcv_model = load_ours(
            input_dim=cfg.data.input_dim,
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
            dim_normalization=cfg.model.mlcv_model.dim_normalization,
            normalization_factor=cfg.model.mlcv_model.normalization_factor,
            transferable=cfg.model.mlcv_model.transferable,
        ).to(device)
    elif method in ["tda","tae", "vde"]:
        mlcv_model = load_baseline(
            model_name=method,
        ).to(device)
    else:
        raise ValueError(f"Method {method} not supported")
    mlcv_model.train()
    for p in mlcv_model.parameters():
        p.requires_grad = True
    if cfg.log.mlcv_model.watch:
        wandb.watch(
            mlcv_model,
            log=cfg.log.mlcv_model.log,
            log_freq=cfg.log.mlcv_model.watch_freq,
        )
    watch_param_list = list(mlcv_model.parameters())
    if cfg.model.score_model.mode == "train":
        watch_param_list = watch_param_list + list(score_model.parameters())
    if cfg.model.mlcv_model.condition_mode == "backbone-both":
        watch_param_list = watch_param_list + list(zero_conv_mlp_pos.parameters()) + list(zero_conv_mlp_orient.parameters())
    elif cfg.model.mlcv_model.condition_mode == "input-control":
        for i in range(8):
            watch_param_list = watch_param_list + list(score_model.model_nn.st_module.encoder.get_submodule(f"zero_conv_mlp_{i}").parameters())
    else:
        watch_param_list = watch_param_list + list(zero_conv_mlp.parameters())
    
    
    # Load training
    mid_t = cfg.model.rollout.mid_t
    N_rollout = cfg.model.rollout.N_rollout
    record_grad_steps = cfg.model.rollout.record_grad_steps
    learning_rate = cfg.model.training.learning_rate
    num_epochs = cfg.model.training.num_epochs
    batch_size = cfg.model.training.batch_size
    optimizer = torch.optim.AdamW(
        watch_param_list,
        lr=learning_rate,
    )
    scheduler_name = cfg.model.training.scheduler.name
    if scheduler_name == "CosineAnnealingWarmUpRestarts":
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=num_epochs,
            T_mult=cfg.model.training.scheduler.T_mult,
            eta_max=cfg.model.training.scheduler.eta_max,
            T_up=cfg.model.training.scheduler.warmup_epochs,
            gamma=cfg.model.training.scheduler.gamma,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")
    
    # Load data
    dataset = TimelagDataset(
        cfg_data = cfg.data,
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    chemgraph = (
        get_context_chemgraph(sequence=cfg.data.sequence)
        .replace(system_id=cfg.data.system_id)
        .to(device)
    )
    num_batches = len(dataloader)

    # Training loop
    training_method = getattr(cfg.model.training, 'method', 'ppft')
    enable_loss_clipping = getattr(cfg.model.training, 'enable_loss_clipping', True)
    max_loss_value = getattr(cfg.model.training, 'max_loss_value', 100.0)
    min_loss_value = getattr(cfg.model.training, 'min_loss_value', 0.0)
    print(f"\n=== TRAINING CONFIGURATION ===")
    print(f"Training method: {training_method}")
    if training_method == 'standard':
        print(f"  - Using efficient standard denoising fine-tuning")
        print(f"  - No rollouts required (much faster)")
    elif training_method == 'ppft':
        print(f"  - Using PPFT method (expensive rollouts)")
        print(f"  - Mid timestep: {mid_t}")
        print(f"  - Rollout steps: {N_rollout}")
        print(f"  - This will be slower but provides structure metrics")
    else:
        raise ValueError(f"Unknown training method: {training_method}. Choose from ['standard', 'ppft']")
    print(f"Loss clipping: {'enabled' if enable_loss_clipping else 'disabled'}")
    if enable_loss_clipping:
        print(f"  - Max loss value: {max_loss_value}")
        print(f"  - Min loss value: {min_loss_value}")
        print(f"  - Purpose: Prevent training instability from extreme losses")
    print(f"==============================\n")
    
    pbar = tqdm(
        range(num_epochs),
        desc=f"Loss: x.xxxxxx",
        total=num_epochs,
    )
    total_clips = 0
    clips_per_epoch = []
    
    for epoch in pbar:
        total_loss = 0
        epoch_clips = 0  # Track clipping events this epoch
        
        # Loss spike debugging
        batch_mlcv_stats = []
        batch_structure_stats = []

        if epoch > cfg.model.score_model.last_training * num_epochs:
            score_model.train()
        
        if cfg.model.mlcv_model.condition_mode == "backbone-both":
            zero_conv_mlp_pos.train()
            zero_conv_mlp_orient.train()
        elif cfg.model.mlcv_model.condition_mode == "input-control":
            for i in range(8):
                score_model.model_nn.st_module.encoder.get_submodule(f"zero_conv_mlp_{i}").train()
        elif cfg.model.mlcv_model.condition_mode in ["input", "latent", "backbone"]:
            zero_conv_mlp.train()

        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            total=len(dataloader),
            leave=False,
        )):
            optimizer.zero_grad()
            current_data = batch["current_data"]
            timelagged_data = batch["timelagged_data"]
            current_orientation = batch["current_orientation"]
            timelagged_orientation = batch["timelagged_orientation"]
            mlcv = mlcv_model(current_data)
            
            if training_method == 'standard':
                loss, loss_metrics = calc_standard_denoising_loss(
                    score_model=score_model,
                    mlcv=mlcv,
                    target=timelagged_data,
                    sdes=sdes,
                    orientation=timelagged_orientation,
                    batch=[chemgraph] * current_data.shape[0],
                    cfg=cfg,
                    condition_mode=cfg.model.mlcv_model.condition_mode,
                    min_t=getattr(cfg.model.training, 'min_t', 1e-4),
                    max_t=getattr(cfg.model.training, 'max_t', 0.99),
                )
                structure_metrics = {}  # No structure metrics for standard denoising
                
            elif training_method == 'ppft':
                loss, structure_metrics = calc_tlc_loss(
                    score_model=score_model,
                    mlcv=mlcv,
                    target=timelagged_data,
                    sdes=sdes,
                    batch=[chemgraph] * current_data.shape[0],
                    n_replications=1,
                    mid_t=mid_t,
                    N_rollout=N_rollout,
                    record_grad_steps=record_grad_steps,
                    condition_mode=cfg.model.mlcv_model.condition_mode,
                    cfg=cfg,
                )
                loss_metrics = {}  # No additional loss metrics for PPFT
                
            else:
                raise ValueError(f"Unknown training method: {training_method}. Choose from ['standard', 'ppft']")
            
            # Apply loss clipping to prevent training instability
            original_loss_value = loss.item()
            loss, was_clipped = clip_loss(
                loss=loss,
                max_loss_value=max_loss_value,
                min_loss_value=min_loss_value,
                enable_clipping=enable_loss_clipping,
                debug=cfg.log.debug
            )
            
            # Track clipping statistics
            if was_clipped:
                epoch_clips += 1
                total_clips += 1
                if cfg.log.debug:
                    print(f"📌 Batch {batch_idx}: Loss clipped from {original_loss_value:.6f} to {loss.item():.6f}")
            
            # Add method-specific metrics to the loss for logging
            if loss_metrics:
                # Store these for logging later
                current_loss_metrics = loss_metrics
            total_loss = total_loss + loss
            
            # Track MLCV characteristics for this batch
            mlcv_stats = {
                'mean': mlcv.mean().item(),
                'std': mlcv.std().item(),
                'min': mlcv.min().item(),
                'max': mlcv.max().item(),
                'has_nan': torch.isnan(mlcv).any().item(),
                'has_inf': torch.isinf(mlcv).any().item(),
            }
            batch_mlcv_stats.append(mlcv_stats)
            
            # Track structure quality for this batch
            if structure_metrics:
                structure_stats = {
                    'ca_mean': structure_metrics.get('ca_sequential_dist_mean', 0),
                    'ca_violations': structure_metrics.get('ca_sequential_dist_violations', 0),
                    'cn_violations': structure_metrics.get('cn_bond_violations', 0),
                }
                batch_structure_stats.append(structure_stats)
            else:
                batch_structure_stats.append({'ca_mean': 0, 'ca_violations': 0, 'cn_violations': 0})
            
            # Collect structure metrics for batch-level logging
            if batch_idx == 0:  # Store metrics from first batch for logging
                current_structure_metrics = structure_metrics
            loss.backward()
            
            # Debug: Check gradient norms before clipping
            if cfg.log.debug and batch_idx == 0:
                total_norm_before = 0
                param_count = 0
                max_grad = 0
                min_grad = float('inf')
                for p in watch_param_list:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_before += param_norm.item() ** 2
                        max_grad = max(max_grad, p.grad.abs().max().item())
                        min_grad = min(min_grad, p.grad.abs().min().item())
                        param_count += 1
                total_norm_before = total_norm_before ** (1. / 2)
                print(f"Gradient norm BEFORE clipping: {total_norm_before:.2e} (max: {max_grad:.2e}, min: {min_grad:.2e})")
            
            # Add gradient clipping to prevent vanishing/exploding gradients
            if cfg.model.training.gradient_clip:
                clip_val = cfg.model.training.gradient_clip_val
                clipped_norm = torch.nn.utils.clip_grad_norm_(watch_param_list, max_norm=clip_val)
            
                if cfg.log.debug and batch_idx == 0:
                    print(f"Gradient clip results: total_norm={clipped_norm:.2e}, clip_val={clip_val}")
                    if clipped_norm > clip_val and clip_val > 0:
                        print(f"⚠️  Gradients were clipped! (norm {clipped_norm:.2e} > {clip_val})")
                    else:
                        print(f"✓ Gradients within bounds (no clipping needed)")
                    
            optimizer.step()
        
        scheduler.step()
        pbar.set_description(f"Loss: {total_loss/num_batches:.6f}")
        
        # Prepare logging data
        log_data = {
            "loss": total_loss/num_batches,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
        }
        wandb.log(log_data, step=epoch)
        
        # Save checkpoint
        if (epoch + 1) % cfg.log.ckpt_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'mlcv_state_dict': mlcv_model.state_dict(),
                'model_state_dict': score_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'condition_mode': cfg.model.mlcv_model.condition_mode,
                'mlcv_dim': cfg.model.mlcv_model.mlcv_dim,
            }
            torch.save(checkpoint, f"model/{cfg.log.date}/checkpoint_{epoch+1}.pt")
        
    # Print loss clipping summary
    total_batches = num_epochs * num_batches
    overall_clip_rate = total_clips / total_batches if total_batches > 0 else 0
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Training completed: {num_epochs} epochs, {total_batches} total batches")
    
    # Add post-processing module to MLCV model based on full dataset
    print(f"\n=== POST-PROCESSING MODULE ===")
    reference_frame_path = None
    if hasattr(cfg.data, 'system_id') and cfg.data.system_id == "CLN025":
        reference_frame_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{cfg.data.system_id}/6bond.pdb"
        print(f"Using reference frame: {reference_frame_path}")
    
    mlcv_model = add_postprocessing_module(
        mlcv_model=mlcv_model,
        dataset=dataset,
        device=device,
        cfg=cfg,
        reference_frame_path=reference_frame_path
    )
    
    # Save final model weights (including post-processing)
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
        'model_state_dict': score_model.state_dict(),
        'condition_mode': cfg.model.mlcv_model.condition_mode,
        'mlcv_dim': cfg.model.mlcv_model.mlcv_dim,
        'loss_clipping_stats': {
            'total_clips': total_clips,
            'total_batches': total_batches,
            'overall_clip_rate': overall_clip_rate,
            'clips_per_epoch': clips_per_epoch,
        }
    }, f"model/{cfg.log.date}/final_model.pt")    
    
    # Save mlcv model (with post-processing)
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
    }, f"model/{cfg.log.date}/mlcv_model.pt")
    mlcv_model.trainer = Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
    dummy_input = torch.randn(1, current_data.shape[1]).to(device)
    traced_model = torch.jit.trace(mlcv_model, dummy_input)
    traced_model.save(f"model/{cfg.log.date}/mlcv_model-jit.pt")
    
    run.finish()
    
            
if __name__ == "__main__":
    main()