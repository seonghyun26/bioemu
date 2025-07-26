import os
import wandb
import hydra
import torch
import yaml
import torch.nn as nn
import mdtraj as md
import math

from datetime import datetime
from pathlib import Path
from torch_geometric.data import Batch
from tqdm import tqdm
from omegaconf import OmegaConf,open_dict

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

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
    cfg: OmegaConf = None,
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
    generated_ca_distances_sum = torch.tensor(0.0, device=device)
    generated_ca_distances_distribution = torch.tensor(0.0, device=device)
    # target_ca_distances_sum = torch.tensor(0.0, device=device)
    
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
        loss = loss + (generated_ca_distances - target[system_idx]).pow(2).mean()
        
        # loss = loss + (generated_ca_seq_distances - target_ca_seq_distances).pow(2).mean()

        # cad_seq_idx = [0, 9, 17, 24, 30, 35, 39, 42, 44]
        seq_idx = torch.arange(n-1)
        generated_ca_seq_distances = generated_ca_pair_distances[seq_idx, seq_idx+1]
        generated_ca_distances_distribution = generated_ca_distances_distribution + generated_ca_seq_distances
        generated_ca_distances_sum = generated_ca_distances_sum + generated_ca_seq_distances.mean()
        
        # target_ca_distances = target[system_idx]
        # target_ca_distances_sum = target_ca_distances_sum + target_ca_distances
    
    loss = loss / num_systems_sampled
    generated_ca_distances_sum = generated_ca_distances_sum / num_systems_sampled
    generated_ca_distances_distribution = generated_ca_distances_distribution / num_systems_sampled
    # plt.hist(generated_ca_distances_sum.cpu().detach().numpy(), bins=10)
    # plt.savefig("generated_ca_distances_sum.png")
    # plt.close()
    # wandb.log({
    #     "generated_ca_distances_sum": wandb.Image("generated_ca_distances_sum.png"),
    # })
    
    # target_ca_distances_sum = target_ca_distances_sum / num_systems_sampled
    # plt.hist(target_ca_distances_sum.cpu().detach().numpy(), bins=10)
    # plt.savefig("target_ca_distances.png")
    # plt.close()
    # wandb.log({
    #     "generated_ca_distances": wandb.Image("generated_ca_distances.png"),
    #     "target_ca_distances": wandb.Image("target_ca_distances.png"),
    # })

    return loss, generated_ca_distances_sum, generated_ca_distances_distribution
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
    
    # NOTE: add zero conv MLP to score model
    if cfg.model.mlcv_model.condition_mode == "backbone":
        hidden_dim = 3
    elif cfg.model.mlcv_model.condition_mode in ["input", "latent"]:
        hidden_dim = 512
    else:
        raise ValueError(f"Condition type {cfg.model.mlcv_model.condition_mode} not supported")
    score_model.model_nn.add_module(f"zero_conv_mlp", nn.Sequential(
        nn.Linear(hidden_dim + cfg.model.mlcv_model.mlcv_dim, hidden_dim),
        nn.ReLU(),
    ))
    zero_conv_mlp = score_model.model_nn.get_submodule("zero_conv_mlp")
    for p in score_model.parameters():
        p.requires_grad = True
    if cfg.log.score_model.watch:
        wandb.watch(
            zero_conv_mlp,
            log=cfg.log.score_model.log,
            log_freq=cfg.log.score_model.watch_freq,
        )
    
    # NOTE: MLCV model
    method = cfg.model.mlcv_model.name
    if method == "ours":
        mlcv_model = load_ours(mlcv_dim=cfg.model.mlcv_model.mlcv_dim).to(device)
    elif method == "tda":
        mlcv_model = load_tda().to(device)
    elif method == "vde":
        mlcv_model = load_vde().to(device)
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
    
    # NOTE: Load training
    repo_dir = Path("~/prj-mlcv/lib/bioemu").expanduser()
    rollout_config_path = repo_dir / cfg.model.sampling.rollout_path
    rollout_config = yaml.safe_load(rollout_config_path.read_text())
    mid_t = rollout_config["mid_t"]
    N_rollout = rollout_config["N_rollout"]
    learning_rate = cfg.model.training.learning_rate
    num_epochs = cfg.model.training.num_epochs
    batch_size = cfg.model.training.batch_size
    if cfg.model.score_model.mode == "eval":
        optimizer = torch.optim.AdamW(
            list(mlcv_model.parameters()) + list(zero_conv_mlp.parameters()),
            lr=learning_rate,
        )
    elif cfg.model.score_model.mode == "train":
        optimizer = torch.optim.AdamW(
            list(mlcv_model.parameters()) + list(score_model.parameters()),
            lr=learning_rate,
        )
    else:
        raise ValueError(f"Score model mode {cfg.model.score_model.mode} not supported")
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
        total_generated_ca_distances_sum = 0
        total_generated_ca_distances_distribution = 0

        if epoch > cfg.model.score_model.last_training * num_epochs:
            score_model.train()
        else:
            score_model.eval()

        for batch in tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            total=len(dataloader),
            leave=False,
        ):
            optimizer.zero_grad()
            current_data = batch["current_data"]
            timelagged_data = batch["timelagged_data"]
            mlcv = mlcv_model(current_data)

            loss, generated_ca_distances_sum, generated_ca_distances_distribution = calc_tlc_loss(
                score_model=score_model,
                mlcv=mlcv,
                target=timelagged_data,
                sdes=sdes,
                batch=[chemgraph] * current_data.shape[0],
                n_replications=1,
                mid_t=mid_t,
                N_rollout=N_rollout,
                condition_mode=cfg.model.mlcv_model.condition_mode,
            )
            total_loss = total_loss + loss
            total_generated_ca_distances_sum = total_generated_ca_distances_sum + generated_ca_distances_sum
            total_generated_ca_distances_distribution = total_generated_ca_distances_distribution + generated_ca_distances_distribution
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        pbar.set_description(f"Loss: {total_loss/num_batches:.6f}")
        wandb.log({
            "loss": total_loss/num_batches,
            "generated_ca_distances_sum": total_generated_ca_distances_sum/num_batches,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            },
            step=epoch,
        )
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'mlcv_state_dict': mlcv_model.state_dict(),
                'model_state_dict': score_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"model/{cfg.log.date}/checkpoint_{epoch+1}.pt")
        
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
    }, f"model/{cfg.log.date}/mlcv.pt")
    
    run.finish()
    
            
if __name__ == "__main__":
    main()