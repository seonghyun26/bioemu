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

from torch.utils.data import DataLoader

from bioemu.chemgraph import ChemGraph
from bioemu.sample import get_context_chemgraph
from bioemu.denoiser import get_score, dpm_solver
from bioemu.sde_lib import SDE
from bioemu.chemgraph import ChemGraph
from bioemu.models import DiGConditionalScoreModel
from bioemu.so3_sde import SO3SDE


from data import *
from model import *


# SINGLE_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_single.npy"
# PAIR_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_pair.npy"
# rollout_config_path = "/home/shpark/prj-mlcv/lib/bioemu/notebook/rollout.yaml"
# OUTPUT_DIR = Path("~/prj-mlcv/lib/bioemu/ppft_example_output").expanduser()
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cln025_alpha_carbon_idx = [4, 25, 46, 60, 72, 87, 101, 108, 122, 146]


def foldedness_by_hbond(
    traj,
    distance_cutoff=0.35,
    bond_number_cutoff=3
):
	"""
	Generate binary labels for folded/unfolded states based at least 3 bonds among eight bonds
	- TYR1T-YR10OT1
	- TYR1T-YR10OT2
	- ASP3N-TYR8O
	- THR6OG1-ASP3O
	- THR6N-ASP3OD1
	- THR6N-ASP3OD2
	- TYR10N-TYR1O


	Args:
		traj (mdtraj): mdtraj trajectory object
		distance_cutoff (float): donor-acceptor distance cutoff in nm (default 0.35 nm = 3.5 amstrong)
		angle_cutoff (float): hydrogen bond angle cutoff in degrees (default 110 deg)
		bond_number_cutoff (int): minimum number of bonds to be considered as folded (default 3)

	Returns:
		labels (np.array): binary array (1: folded, 0: unfolded)
	"""
	# TYR1N-YR10OT1
	donor_idx = traj.topology.select('residue 1 and name N')[0] # Tyr1:N
	acceptor_idx = traj.topology.select('residue 10 and name O')[0]   # Tyr10:OT1
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_O1 = ((distance[:,0] < distance_cutoff)).astype(int)
	label_O2 = ((distance[:,0] < distance_cutoff)).astype(int) 
	label_O3 = ((distance[:,0] < distance_cutoff)).astype(int)
	label_TYR1N_TYR10OT1 = label_O1 | label_O2 | label_O3


	# TYR1N-YR10OT2
	donor_idx = traj.topology.select('residue 1 and name N')[0] # Tyr1:N
	acceptor_idx = traj.topology.select('residue 10 and name OXT')[0]   # Tyr10:OT2
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_O1 = ((distance[:,0] < distance_cutoff)).astype(int)
	label_O2 = ((distance[:,0] < distance_cutoff)).astype(int)
	label_O3 = ((distance[:,0] < distance_cutoff)).astype(int)
	label_TYR1N_TYR10OT2 = label_O1 | label_O2 | label_O3


	# ASP3N-TYR8O
	donor_idx = traj.topology.select('residue 3 and name N')[0]
	acceptor_idx = traj.topology.select('residue 8 and name O')[0]
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_ASP3N_TYR8O = ((distance[:,0] < distance_cutoff)).astype(int)
 
 
	# THR6OG1-ASP3O
	donor_idx = traj.topology.select('residue 6 and name OG1')[0]
	acceptor_idx = traj.topology.select('residue 3 and name O')[0]
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_THR6OG1_ASP3O = ((distance[:,0] < distance_cutoff)).astype(int)
 
 
	# THR6N-ASP3OD1
	donor_idx = traj.topology.select('residue 6 and name N')[0]
	acceptor_idx = traj.topology.select('residue 3 and name OD1')[0]
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_THR6N_ASP3OD1 = ((distance[:,0] < distance_cutoff)).astype(int)
 
	# THR6N-ASP3OD2
	donor_idx = traj.topology.select('residue 6 and name N')[0]
	acceptor_idx = traj.topology.select('residue 3 and name OD2')[0]
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_THR6N_ASP3OD2 = ((distance[:,0] < distance_cutoff)).astype(int)
 
 
	# GLY7N-ASP3O
	donor_idx = traj.topology.select('residue 7 and name N')[0]
	acceptor_idx = traj.topology.select('residue 3 and name O')[0]
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_GLY7N_ASP3O = ((distance[:,0] < distance_cutoff)).astype(int)
 

	# TYR10N-TYR1O
	donor_idx = traj.topology.select('residue 10 and name N')[0] 
	acceptor_idx = traj.topology.select('residue 1 and name O')[0] 
	distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
	label_TYR10N_TYR1O = ((distance[:,0] < distance_cutoff)).astype(int)




	# ASP3OD_THR6OG1_ASP3N_THR8O
	bond_sum = label_TYR1N_TYR10OT1 + label_TYR1N_TYR10OT2 + label_ASP3N_TYR8O + label_THR6OG1_ASP3O \
		+ label_THR6N_ASP3OD1 + label_THR6N_ASP3OD2 + label_GLY7N_ASP3O + label_TYR10N_TYR1O
	labels = bond_sum >= bond_number_cutoff

	return labels, bond_sum

    
    
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


# def unphysical_loss(
#     x0_pos: torch.Tensor,
#     max_ca_seq_distance: float = 4.5,
#     max_cn_seq_distance: float = 2.0,
#     clash_distance: float = 1.0,
# ) -> torch.Tensor:
#     # pdb_path = "/home/shpark/prj-mlcv/lib/bioemu/topology-backbone.pdb"
#     # with open(pdb_path, "r") as f:
#     #     pdb_lines = f.readlines()
#     # traj = md.load_pdb(pdb_path)
#     # traj.xyz = x0_pos.clone().cpu().detach().numpy()

#     # ca_indices = [4, 25, 46, 60, 72, 87, 101, 108, 122, 146]
#     # c_indices = [5, 26, 47, 61, 73, 88, 102, 109, 123, 147]
#     # n_indices = [6, 27, 48, 62, 74, 89, 103, 110, 124, 148]
#     # max_ca_seq_distance = 1.54
#     # max_cn_seq_distance = 1.33
#     # clash_distance = 1.9
    
#     n = x0_pos.shape[0]
#     diffs = x0_pos.unsqueeze(1) - x0_pos.unsqueeze(0)  # shape: (n, n, 3)
#     dists = torch.norm(diffs, dim=-1)  # shape: (n, n
#     mask = ~torch.eye(n, dtype=torch.bool, device=x0_pos.device)
#     contact_loss = torch.nn.functional.relu(dists - max_ca_seq_distance)
#     loss_ca = contact_loss[mask].pow(2).mean()

#     # # C-N peptide bond distance
#     # cn_dists = torch.norm(
#     #     xyz[:, c_indices] - xyz[:, n_indices], dim=-1
#     # )
#     # loss_cn = torch.nn.functional.relu(cn_dists - max_cn_seq_distance).pow(2).mean()
#     loss_cn = 0

#     # # Clash penalty
#     # # Compute full distance matrix (B, N, N)
#     # diff = xyz[:, :, None, :] - xyz[:, None, :, :]
#     # dists = torch.norm(diff, dim=-1)

#     # # Mask out intra-residue distances
#     # same_res = torch.eye(xyz.shape[0], dtype=torch.bool, device = device)
#     # clash_mask = torch.logical_not(same_res).to(device)

#     # # Apply clash penalty
#     # clash_penalty = torch.nn.functional.relu(clash_distance - dists)
#     # clash_penalty = clash_penalty * clash_mask  # ignore same-residue
#     # loss_clash = clash_penalty.pow(2).mean()
#     loss_clash = 0
    
#     return loss_ca, loss_cn, loss_clash


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
    )
    num_systems_sampled = len(batch)
    
    # =================================================================
    # STRUCTURE QUALITY MONITORING: Compute CA and C-N distances
    # =================================================================
    structure_metrics = {}
    
    try:
        # Load reference structure to get atom indices
        pdb = md.load_pdb(f"/home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres_backbone.pdb")
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
        generated_ca_distance = torch.empty(num_systems_sampled, n_replications, 45, device=device)
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
        pdb = md.load_pdb(f"/home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres_backbone.pdb")
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
        record_grad_steps=record_grad_steps,  # Enable gradients for all steps
    )

    # Predict clean x (x0) from x_mid in a single jump.
    # This step is always with gradient.
    mid_t_expanded = torch.full((batch_size,), mid_t, device=device)
    score_mid_t = get_score(
        batch=x_mid, sdes=sdes, t=mid_t_expanded, score_model=score_model, mlcv=mlcv,
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
    if init_mode == "zero":
        zero_linear = nn.Linear(hidden_dim + mlcv_dim, hidden_dim)
        nn.init.zeros_(zero_linear.weight)  # Keep bias zero
        nn.init.zeros_(zero_linear.bias)  # Keep bias zero
        zero_mlp = nn.Sequential(zero_linear)
    elif init_mode == "rand":
        zero_linear = nn.Linear(hidden_dim + mlcv_dim, hidden_dim)
        nn.init.normal_(zero_linear.weight, mean=0.0, std=1e-4)  # Very small random weights
        nn.init.zeros_(zero_linear.bias)  # Keep bias zero
        zero_mlp = nn.Sequential(zero_linear, nn.ReLU())
    else:
        raise ValueError(f"Invalid initialization method: {init_mode}")
    
    return zero_mlp


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
    
    # NOTE: Set logging
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
    
    # NOTE: score model
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
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])
    if cfg.model.score_model.mode == "train":
        score_model.train()
    elif cfg.model.score_model.mode == "eval":
        score_model.eval()
    else:
        raise ValueError(f"Score model mode {cfg.model.score_model.mode} not supported")
    # if cfg.log.debug:
        # score_model.model_nn._debug_conditioning = True
        # score_model.model_nn.st_module._debug_conditioning = True
        
    
    # NOTE: add zero conv MLP to score model
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
        
    elif cfg.model.mlcv_model.condition_mode == "input-control":
        for i in range(8):
            zero_mlp = _add_zero_conv_mlp(
                init_mode=cfg.model.score_model.init,
                hidden_dim=hidden_dim,
                mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
            )
            zero_mlp.train()
            score_model.model_nn.st_module.encoder.add_module(f"zero_conv_mlp_{i}", zero_mlp)
    
    elif cfg.model.mlcv_model.condition_mode == "backbone-both":
        zero_mlp_pos = _add_zero_conv_mlp(
            init_mode=cfg.model.score_model.init,
            hidden_dim=hidden_dim,
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
        )
        zero_mlp_pos.train()
        score_model.model_nn.add_module(f"zero_conv_mlp_pos", zero_mlp_pos)
        zero_conv_mlp_pos = score_model.model_nn.get_submodule("zero_conv_mlp_pos")
        
        zero_mlp_orient = _add_zero_conv_mlp(
            init_mode=cfg.model.score_model.init,
            hidden_dim=hidden_dim,
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
        )
        zero_mlp_orient.train()
        score_model.model_nn.add_module(f"zero_conv_mlp_orient", zero_mlp_orient)
        zero_conv_mlp_orient = score_model.model_nn.get_submodule("zero_conv_mlp_orient")
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
    
    
    # NOTE: MLCV model
    method = cfg.model.mlcv_model.name
    if method == "ours":
        mlcv_model = load_ours(
            mlcv_dim=cfg.model.mlcv_model.mlcv_dim,
            dim_normalization=cfg.model.mlcv_model.dim_normalization,
            normalization_factor=cfg.model.mlcv_model.normalization_factor,
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
    
    
    # NOTE: Load training
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
    
    # NOTE: Load data
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

    # NOTE: Training loop
    pbar = tqdm(
        range(num_epochs),
        desc=f"Loss: x.xxxxxx",
        total=num_epochs,
    )
    for epoch in pbar:
        total_loss = 0
        current_structure_metrics = {}  # Initialize structure metrics for this epoch

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
            mlcv = mlcv_model(current_data)
            
            if cfg.log.debug_mlcv and batch_idx == 0:  # Only debug first batch to avoid spam
                print(f"=== CONDITIONING EFFECT DEBUG ===")
                print(f"MLCV values: {mlcv.flatten()[:10].detach().cpu().numpy()}")
                
                # Test: Run the same input with and without MLCV to see if outputs differ
                with torch.no_grad():
                    loss_with_mlcv, _ = calc_tlc_loss(
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
                    
                    # Create zero MLCV for comparison
                    mlcv_zero = torch.zeros_like(mlcv)
                    loss_without_mlcv, _ = calc_tlc_loss(
                        score_model=score_model,
                        mlcv=mlcv_zero,
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
                    
                    # Also test with random MLCV
                    mlcv_random = torch.randn_like(mlcv) * 0.1
                    loss_with_random_mlcv, _ = calc_tlc_loss(
                        score_model=score_model,
                        mlcv=mlcv_random,
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
                    
                print(f"Loss with real MLCV:   {loss_with_mlcv.item():.6f}")
                print(f"Loss with zero MLCV:   {loss_without_mlcv.item():.6f}")
                print(f"Loss with random MLCV: {loss_with_random_mlcv.item():.6f}")
                print(f"Real vs Zero effect:   {abs(loss_with_mlcv.item() - loss_without_mlcv.item()):.6f}")
                print(f"Real vs Random effect: {abs(loss_with_mlcv.item() - loss_with_random_mlcv.item()):.6f}")
                
                if abs(loss_with_mlcv.item() - loss_without_mlcv.item()) < 1e-6:
                    print("❌ CRITICAL: Conditioning has NO effect on loss!")
                else:
                    print("✅ Conditioning affects loss (good!)")
                print(f"=== END DEBUG ===")

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
            total_loss = total_loss + loss
            
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
            if hasattr(cfg.model.training, 'gradient_clip_val') and cfg.model.training.gradient_clip_val > 0:
                clip_val = cfg.model.training.gradient_clip_val
            else:
                clip_val = 1.0  # Default clip value
            
            # Clip gradients for all parameters being optimized
            clipped_norm = torch.nn.utils.clip_grad_norm_(watch_param_list, max_norm=clip_val)
            
            # Debug: Check gradient clipping results
            if cfg.log.debug and batch_idx == 0:
                print(f"Gradient clip results: total_norm={clipped_norm:.2e}, clip_val={clip_val}")
                if clipped_norm > clip_val:
                    print(f"⚠️  Gradients were clipped! (norm {clipped_norm:.2e} > {clip_val})")
                else:
                    print(f"✓ Gradients within bounds (no clipping needed)")
            
            # Optional gradient flow debugging (add debug flag to config to enable)
            if cfg.log.debug:
                zero_conv_modules = []
                if cfg.model.mlcv_model.condition_mode == "backbone-both":
                    zero_conv_modules.extend([zero_conv_mlp_pos, zero_conv_mlp_orient])
                elif cfg.model.mlcv_model.condition_mode == "input-control":
                    for i in range(8):
                        zero_conv_modules.append(score_model.model_nn.st_module.encoder.get_submodule(f"zero_conv_mlp_{i}"))
                elif cfg.model.mlcv_model.condition_mode in ["input", "latent", "backbone"]:
                    zero_conv_modules.append(zero_conv_mlp)
                
                for i, module in enumerate(zero_conv_modules):
                    # Check gradients more carefully
                    grad_info = []
                    for name, param in module.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_max = param.grad.abs().max().item()
                            grad_info.append(f"{name}: norm={grad_norm:.2e}, max={grad_max:.2e}")
                        else:
                            grad_info.append(f"{name}: grad=None")
                    
                    print(f"Zero conv module {i} detailed gradients: {grad_info}")
                    has_meaningful_grad = any(p.grad is not None and p.grad.abs().sum() > 1e-8 for p in module.parameters())
                    if not has_meaningful_grad:
                        print(f"WARNING: Zero conv module {i} has no meaningful gradients!")
                    else:
                        print(f"✓ Zero conv module {i} has gradients!")
                
            optimizer.step()
        
        scheduler.step()
        pbar.set_description(f"Loss: {total_loss/num_batches:.6f}")
        
        # Prepare logging data
        log_data = {
            "loss": total_loss/num_batches,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
        }
        
        # Add structure quality metrics if available
        if 'current_structure_metrics' in locals() and current_structure_metrics:
            for key, value in current_structure_metrics.items():
                log_data[f"structure/{key}"] = value
            
            # Additional logging for structure quality monitoring
            if 'ca_sequential_dist_mean' in current_structure_metrics:
                ca_mean = current_structure_metrics['ca_sequential_dist_mean']
                ca_violations = current_structure_metrics.get('ca_sequential_dist_violations', 0)
                log_data['structure/ca_health_score'] = max(0, 1 - ca_violations)  # Health score: 1 - violation_rate
                log_data['structure/ca_realistic'] = 1 if 0.3 <= ca_mean <= 0.4 else 0  # 1 if CA distances are realistic
        
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
        
    # Save final model weights
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
        'model_state_dict': score_model.state_dict(),
        'condition_mode': cfg.model.mlcv_model.condition_mode,
        'mlcv_dim': cfg.model.mlcv_model.mlcv_dim,
    }, f"model/{cfg.log.date}/final.pt")    
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
    }, f"model/{cfg.log.date}/mlcv_model.pt")
    
    run.finish()
    
            
if __name__ == "__main__":
    main()