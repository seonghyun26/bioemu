import os
import wandb
import hydra
import numpy as np
import torch
import yaml
import lightning
import argparse
import torch.nn as nn
import mdtraj as md
import math
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Optional
from itertools import combinations

from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from bioemu.chemgraph import ChemGraph
from bioemu.sample import get_context_chemgraph
from bioemu.denoiser import get_score, dpm_solver
from bioemu.sde_lib import SDE
from bioemu.chemgraph import ChemGraph
from bioemu.models import EVOFORMER_EDGE_DIM, EVOFORMER_NODE_DIM
from bioemu.models import DiGConditionalScoreModel
from bioemu.structure_module import SAEncoderLayer
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.so3_sde import SO3SDE

from mlcolvar.cvs import BaseCV, DeepTDA, VariationalAutoEncoderCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.loss.elbo import elbo_gaussians_loss
from mlcolvar.core.transform import Transform


# SINGLE_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_single.npy"
# PAIR_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_pair.npy"
# rollout_config_path = "/home/shpark/prj-mlcv/lib/bioemu/notebook/rollout.yaml"
OUTPUT_DIR = Path("~/prj-mlcv/lib/bioemu/ppft_example_output").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
repo_dir = Path("~/prj-mlcv/lib/bioemu").expanduser()
rollout_config_path = repo_dir / "notebook" / "rollout.yaml"

seed = 0
WANDB_WATCH_FREQ = 10
torch.manual_seed(seed)


cln025_alpha_carbon_idx = [4, 25, 46, 60, 72, 87, 101, 108, 122, 146]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="YYDPETGTWY")
    parser.add_argument("--method", type=str, default="ours")
    parser.add_argument("--learning_rate", type=float, default=1e-10)
    parser.add_argument("--eta_max", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--time_lag", type=int, default=100)
    parser.add_argument("--date", type=str, default="debug")
    parser.add_argument("--score_model_mode", type=str, default="eval")
    parser.add_argument("--mlcv_dim", type=int, default=2)
    parser.add_argument("--last_training", type=float, default=1)
    parser.add_argument("--tags", nargs='+',type=str, default=["pilot"])
    parser.add_argument("--condition_mode", type=str, default="latent")
    parser.add_argument("--physical_loss_weight", type=float, default=0.0)
    parser.add_argument("--param_watch", type=str2bool, default=False)
    
    return parser


class DIM_NORMALIZATION(Transform):
    def __init__(
        self,
        feature_dim = 1
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
        
    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=-1)
        return x
    

class MLCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]

    def __init__(
        self,
        mlcv_dim: int,
        encoder_layers: list,
        options: dict = None,
        **kwargs,
    ):
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
        self.postprocessing = DIM_NORMALIZATION(mlcv_dim)


class VDELoss(torch.nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        z_t: torch.Tensor,
        z_t_tau: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elbo_loss = elbo_gaussians_loss(target, output, mean, log_variance, weights)
        auto_correlation_loss = 0
        
        z_t_mean = z_t.mean(dim=0)
        z_t_tau_mean = z_t_tau.mean(dim=0)
        z_t_centered = z_t - z_t_mean.repeat(z_t.shape[0], 1)
        z_t_tau_centered = z_t_tau - z_t_tau_mean.repeat(z_t_tau.shape[0], 1)
        
        # auto_correlation_loss = - (z_t_centered @ z_t_tau_centered.T)[torch.eye(z_t.shape[0], dtype=torch.bool, device = z_t.device)].mean()
        # auto_correlation_loss = auto_correlation_loss / (z_t.std(dim=0).T @ z_t_tau.std(dim=0))
        ac_num = z_t_centered.reshape(1, -1) @ z_t_tau_centered.reshape(-1, 1)
        ac_den = z_t_centered.norm(2) * z_t_tau_centered.norm(2)
        auto_correlation_loss = - ac_num / ac_den
        
        return elbo_loss, auto_correlation_loss
        
        
class VariationalDynamicsEncoder(VariationalAutoEncoderCV):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # =======   LOSS  =======
        # ELBO loss function when latent space and reconstruction distributions are Gaussians.
        self.loss_fn = VDELoss()
        self.optimizer = Adam(self.parameters(), lr=1e-4)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)
    
    def training_step(
        self,
        train_batch, 
        batch_idx
    ):
        x = train_batch["data"]
        input = x
        loss_kwargs = {}
        if "weights" in train_batch:
            loss_kwargs["weights"] = train_batch["weights"]

        # Encode/decode.
        mean, log_variance, x_hat = self.encode_decode(x)

        # Reference output (compare with a 'target' key if any, otherwise with input 'data')
        if "target" in train_batch:
            x_ref = train_batch["target"]
        else:
            x_ref = x
        
        # Values for autocorrealtion loss
        if self.norm_in is not None:
            input_normalized = self.norm_in(input)
            x_ref_normalized = self.norm_in(x_ref)
        z_t = self.encoder(input_normalized)
        z_t_tau = self.encoder(x_ref_normalized)
        
        # Loss function.
        elbo_loss, auto_correlation_loss = self.loss_fn(
            x_ref, x_hat, mean, log_variance,
            z_t, z_t_tau,
            **loss_kwargs
        )

        # Log.
        name = "train" if self.training else "valid"
        self.log(f"{name}_elbo_loss", elbo_loss, on_epoch=True)
        self.log(f"{name}_auto_correlation_loss", auto_correlation_loss, on_epoch=True)
        self.log(f"{name}_loss", elbo_loss + auto_correlation_loss, on_epoch=True)

        return elbo_loss + auto_correlation_loss


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


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


def load_data(
    simulation_idx=0,
    # time_lag=5,
):
    # cln025_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-{simulation_idx}-protein/CLN025-{simulation_idx}-CAdistance-mdtraj.pt"
    # cln025_pos_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-{simulation_idx}-protein/CLN025-{simulation_idx}-CAdistance-mdtraj.pt"
    current_data_path = "/home/shpark/prj-mlcv/lib/DESRES/dataset/CLN025-5k-current-cad.pt"
    timelag_data_path = "/home/shpark/prj-mlcv/lib/DESRES/dataset/CLN025-5k-timelagged-cad.pt"
    # timelag_data_path = "/home/shpark/prj-mlcv/lib/DESRES/dataset/CLN025-5k-timelagged-label.pt"
    current_data = torch.load(current_data_path)
    timelag_data = torch.load(timelag_data_path)
    
    return current_data, timelag_data
    
    
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


def unphysical_loss(
    x0_pos: torch.Tensor,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
) -> torch.Tensor:
    # pdb_path = "/home/shpark/prj-mlcv/lib/bioemu/topology-backbone.pdb"
    # with open(pdb_path, "r") as f:
    #     pdb_lines = f.readlines()
    # traj = md.load_pdb(pdb_path)
    # traj.xyz = x0_pos.clone().cpu().detach().numpy()

    # ca_indices = [4, 25, 46, 60, 72, 87, 101, 108, 122, 146]
    # c_indices = [5, 26, 47, 61, 73, 88, 102, 109, 123, 147]
    # n_indices = [6, 27, 48, 62, 74, 89, 103, 110, 124, 148]
    # max_ca_seq_distance = 1.54
    # max_cn_seq_distance = 1.33
    # clash_distance = 1.9
    
    n = x0_pos.shape[0]
    diffs = x0_pos.unsqueeze(1) - x0_pos.unsqueeze(0)  # shape: (n, n, 3)
    dists = torch.norm(diffs, dim=-1)  # shape: (n, n
    mask = ~torch.eye(n, dtype=torch.bool, device=x0_pos.device)
    contact_loss = torch.nn.functional.relu(dists - max_ca_seq_distance)
    loss_ca = contact_loss[mask].pow(2).mean()

    # # C-N peptide bond distance
    # cn_dists = torch.norm(
    #     xyz[:, c_indices] - xyz[:, n_indices], dim=-1
    # )
    # loss_cn = torch.nn.functional.relu(cn_dists - max_cn_seq_distance).pow(2).mean()
    loss_cn = 0

    # # Clash penalty
    # # Compute full distance matrix (B, N, N)
    # diff = xyz[:, :, None, :] - xyz[:, None, :, :]
    # dists = torch.norm(diff, dim=-1)

    # # Mask out intra-residue distances
    # same_res = torch.eye(xyz.shape[0], dtype=torch.bool, device = device)
    # clash_mask = torch.logical_not(same_res).to(device)

    # # Apply clash penalty
    # clash_penalty = torch.nn.functional.relu(clash_distance - dists)
    # clash_penalty = clash_penalty * clash_mask  # ignore same-residue
    # loss_clash = clash_penalty.pow(2).mean()
    loss_clash = 0
    
    return loss_ca, loss_cn, loss_clash


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
    loss_func = None,
    condition_mode: str = "none",
) -> torch.Tensor:
    device = batch[0].pos.device
    assert isinstance(batch, list)  # Not a Batch!

    x_in = Batch.from_data_list(batch * n_replications)
    x0 = _rollout(
        batch=x_in,
        sdes=sdes,
        score_model=score_model,
        mid_t=mid_t,
        N_rollout=N_rollout,
        device=device,
        mlcv=mlcv,
        condition_mode=condition_mode,
    )
    num_systems_sampled = len(batch)

    loss = torch.tensor(0.0, device=device)
    generated_ca_distances_sum  = torch.tensor(0.0, device=device)
    target_ca_distances_sum = torch.tensor(0.0, device=device)
    dummy_traj = md.load(
        "/home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres.pdb"
    )
    ca_resid_pair = np.array(
        [(a.index, b.index) for a, b in combinations(list(dummy_traj.topology.residues), 2)]
    )
    
    for system_idx in range(num_systems_sampled):
        single_system_batch: list[ChemGraph] = x0.get_example(system_idx)
        single_system_batch_pos = single_system_batch.pos
        # target_pos = target[i][cln025_alpha_carbon_idx]
        # loss = loss + kabsch_rmsd(single_system_batch.pos, target_pos)
        
        # NOTE: compute loss between CA distances
        generated_ca_pair_distances = torch.cdist(single_system_batch_pos, single_system_batch_pos, p=2)
        n = generated_ca_pair_distances.shape[0]
        i, j = torch.triu_indices(n, n, offset=1)
        generated_ca_distances = generated_ca_pair_distances[i, j]
        target_ca_distances = target[system_idx]
        seq_idx = torch.arange(n-1)
        generated_ca_seq_distances = generated_ca_pair_distances[seq_idx, seq_idx+1]
        cad_seq_idx = [0, 9, 17, 24, 30, 35, 39, 42, 44]
        target_ca_seq_distances = target[system_idx][cad_seq_idx]

        loss = loss + (generated_ca_seq_distances - target_ca_seq_distances).pow(2).sum()
        # ca_seq_loss = loss_func(generated_ca_seq_distances, target_ca_seq_distances)
        # ca_dist_loss = loss_func(generated_ca_distances, target_ca_distances)
        # loss = loss + 10 * ca_seq_loss + ca_dist_loss
        # generated_ca_distances_sum = generated_ca_distances_sum + generated_ca_distances
        # target_ca_distances_sum = target_ca_distances_sum + target_ca_distances
    
    loss = loss / num_systems_sampled
    # generated_ca_distances_sum = generated_ca_distances_sum / num_systems_sampled
    # target_ca_distances_sum = target_ca_distances_sum / num_systems_sampled
    # ca_seq_distances = 0
    # tar_seq_distances_sum = 0
    # plt.hist(generated_ca_distances_sum.cpu().detach().numpy(), bins=10)
    # plt.savefig("generated_ca_distances.png")
    # plt.close()
    # plt.hist(target_ca_distances_sum.cpu().detach().numpy(), bins=10)
    # plt.savefig("target_ca_distances.png")
    # plt.close()
    # wandb.log({
    #     "generated_ca_distances": wandb.Image("generated_ca_distances.png"),
    #     "target_ca_distances": wandb.Image("target_ca_distances.png"),
    # })

    return loss
    # return loss, generated_ca_distances_sum, target_ca_distances_sum


def _rollout(
    batch: Batch,
    sdes: dict[str, SDE],
    score_model,
    mid_t: float,
    N_rollout: int,
    device: torch.device,
    mlcv: torch.Tensor,
    condition_mode: str = "none",
):
    """Fast rollout to get a sampled structure in a small number of steps.
    Note that in the last step, only the positions are calculated, and not the orientations,
    because the orientations are not used to compute foldedness.
    """
    batch_size = batch.num_graphs
    
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
    

def load_ours(mlcv_dim: int = 2):
    encoder_layers = [45, 30, 30, mlcv_dim]
    options = {
        "encoder": {
            "activation": "tanh",
            "dropout": [0.1, 0.1, 0.1]
        },
        "norm_in": {
        },
    }
    mlcv_model = MLCV(
        mlcv_dim = mlcv_dim,
        encoder_layers = encoder_layers,
        options = options
    )
    mlcv_model.train()
    
    return mlcv_model

def load_tda():
    model = torch.jit.load("/home/shpark/prj-mlcv/lib/bioemu/model/tda-jit.pt")
    model.eval()
    return model

def load_vde():
    model = torch.jit.load("/home/shpark/prj-mlcv/lib/bioemu/model/vde-jit.pt")
    model.eval()
    return model

class CaDistanceDataset(Dataset):
    def __init__(
        self,
        current_data,
        timelag_data,
        device,
    ):
        self.current_data = current_data.to(device)
        self.timelag_data = timelag_data.to(device)

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        return {
            "current_data": self.current_data[idx],
            "timelagged_data": self.timelag_data[idx]
        }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = init_parser()
    args = parser.parse_args()
    sequence = args.sequence
    method = args.method
    if args.date is None:
        args.date = datetime.now().strftime("%m%d_%H%M%S")
    if args.date != "debug":
        os.makedirs(f"model/{args.date}")
    mlcv_dim = args.mlcv_dim
    
    run = wandb.init(
        project="bioemu-ctrl",
        entity="eddy26",
        tags=args.tags,
        config=args,
    )
    
    # Load score model
    ckpt_path = "./model/checkpoint.ckpt"
    cfg_path = "./model/config.yaml"
    with open(cfg_path) as f:
        model_config = yaml.safe_load(f)
    model_config["score_model"]["condition_mode"] = args.condition_mode
    wandb.config.update(model_config)
    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])
    if args.score_model_mode == "train":
        score_model.train()
    elif args.score_model_mode == "eval":
        score_model.eval()
    else:
        raise ValueError(f"Score model mode {args.score_model_mode} not supported")
    
    # NOTE:, add zero conv MLP to score model
    # hidden_dim = 512
    if args.condition_mode == "backbone":
        hidden_dim = 3
    elif args.condition_mode == "latent" or args.condition_mode == "input":
        hidden_dim = 512
    else:
        raise ValueError(f"Condition type {args.condition_mode} not supported")
    score_model.model_nn.add_module(f"zero_conv_mlp", nn.Sequential(
        nn.Linear(hidden_dim + mlcv_dim, hidden_dim),
        nn.ReLU(),
        # nn.Linear(hidden_dim, hidden_dim),
    ))
    for p in score_model.parameters():
        p.requires_grad = True
    if args.param_watch:
        # wandb.watch(
        #     score_model,
        #     log="all",
        #     log_freq=WANDB_WATCH_FREQ,
        # )
        wandb.watch(
            score_model.model_nn.get_submodule("zero_conv_mlp"),
            log="all",
            log_freq=WANDB_WATCH_FREQ,
        )
    
    # Load MLCV model
    if method == "ours" or method == "debug":
        mlcv_model = load_ours(mlcv_dim=mlcv_dim).to(device)
    elif method == "tda":
        mlcv_model = load_tda().to(device)
    elif method == "vde":
        mlcv_model = load_vde().to(device)
    else:
        raise ValueError(f"Method {method} not supported")
    mlcv_model.train()
    # for p in mlcv_model.parameters():
    #     p.requires_grad = True
    print(mlcv_model)
    if args.param_watch:
        wandb.watch(
            mlcv_model,
            log="all",
            log_freq=WANDB_WATCH_FREQ,
        )
    
    # Load config
    rollout_config = yaml.safe_load(rollout_config_path.read_text())
    mid_t = rollout_config["mid_t"]
    N_rollout = rollout_config["N_rollout"]
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    if args.score_model_mode == "eval":
        optimizer = torch.optim.AdamW(list(mlcv_model.parameters()) + list(score_model.model_nn.get_submodule("zero_conv_mlp").parameters()), lr=learning_rate)
    elif args.score_model_mode == "train":
        optimizer = torch.optim.AdamW(list(mlcv_model.parameters()) + list(score_model.parameters()), lr=learning_rate)
    else:
        raise ValueError(f"Score model mode {args.score_model_mode} not supported")
    # optimizer = torch.optim.AdamW(mlcv_model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_max=args.eta_max, T_up=args.warmup_epochs, gamma=0.5)
    
    # Load simulation data
    time_lag = args.time_lag
    current_data, timelag_data = load_data()
    chemgraph = (
        get_context_chemgraph(sequence=sequence)
        .replace(system_id="CLN025")
        .to(device)
    )
    dataset = CaDistanceDataset(
        current_data=current_data,
        timelag_data=timelag_data,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(dataloader)

    # Training loop
    mse_loss = nn.MSELoss()
    pbar = tqdm(
        range(num_epochs),
        desc=f"Loss: x.xxxxxx",
        total=num_epochs,
    )
    for epoch in pbar:
        total_loss = 0
        total_ca_seq_distances = 0
        total_tar_seq_distances_sum = 0
        if epoch > args.last_training * num_epochs:
            score_model.train()
        else:
            score_model.eval()

        for batch in tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=len(dataloader),
            leave=False,
        ):
            optimizer.zero_grad()
            current_data = batch["current_data"]
            timelagged_data = batch["timelagged_data"]
            mlcv = mlcv_model(current_data)

            # loss, ca_seq_distances, tar_seq_distances_sum = calc_tlc_loss(
            loss = calc_tlc_loss(
                score_model=score_model,
                mlcv=mlcv,
                target=timelagged_data,
                sdes=sdes,
                batch=[chemgraph] * current_data.shape[0],
                n_replications=1,
                mid_t=mid_t,
                N_rollout=N_rollout,
                condition_mode=args.condition_mode,
                loss_func=mse_loss,
            )
            total_loss = total_loss + loss
            # total_ca_seq_distances = total_ca_seq_distances + ca_seq_distances
            # total_tar_seq_distances_sum = total_tar_seq_distances_sum + tar_seq_distances_sum
            loss.backward()
            optimizer.step()
            
        
        scheduler.step()
        # Print epoch statistics
        pbar.set_description(f"Loss: {total_loss/num_batches:.6f}")
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "loss": total_loss/num_batches,
            # "ca_seq_distances": total_ca_seq_distances/num_batches,
            # "ca_seq_distances_mean": total_ca_seq_distances.mean()/num_batches,
            # "tar_seq_distances": total_tar_seq_distances_sum/num_batches,
            # "tar_seq_distances_mean": total_tar_seq_distances_sum.mean()/num_batches,
            "lr": current_lr,
            "epoch": epoch,
            "score_model_mode_status": score_model.training,
            },
            step=epoch,
        )
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'mlcv_state_dict': mlcv_model.state_dict(),
                'model_state_dict': score_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"model/{args.date}/checkpoint_{epoch+1}.pt")
        
            # print(f"MLCV: {mlcv}")
        
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
    }, f"model/{args.date}/mlcv.pt")
    
    run.finish()
    
            
if __name__ == "__main__":
    main()