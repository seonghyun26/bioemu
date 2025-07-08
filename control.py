import logging
import os
import typing
import wandb
import hydra
import numpy as np
import torch
import yaml
import lightning
import argparse
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Optional
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader

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


# SINGLE_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_single.npy"
# PAIR_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_pair.npy"
# rollout_config_path = "/home/shpark/prj-mlcv/lib/bioemu/notebook/rollout.yaml"
OUTPUT_DIR = Path("~/prj-mlcv/lib/bioemu/ppft_example_output").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
repo_dir = Path("~/prj-mlcv/lib/bioemu").expanduser()
rollout_config_path = repo_dir / "notebook" / "rollout.yaml"

seed = 0
torch.manual_seed(seed)
date_str = datetime.now().strftime("%m%d_%H%M")
os.makedirs(f"model/{date_str}")


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="GYDPETGTWG")
    parser.add_argument("--method", type=str, default="ours")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--time_lag", type=int, default=5)
    parser.add_argument("--date", type=str, default=date_str)
    parser.add_argument("--score_model_mode", type=str, default="train")
    parser.add_argument("--mlcv_dim", type=int, default=2)
    parser.add_argument("--last_training", type=float, default=1)
    parser.add_argument("--tags", nargs='+',type=str, default=["pilot"])
    parser.add_argument("--condition_type", type=str, default="latent")
    
    return parser


class MLCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]

    def __init__(
        self,
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

def load_data(
    simulation_idx=0,
    # time_lag=5,
):
    cln025_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-{simulation_idx}-protein/CLN025-{simulation_idx}-CAdistance-switch.pt"
    cln025_pos_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-{simulation_idx}-protein/CLN025-{simulation_idx}-coordinates.pt"
    ca_distance_data = torch.load(cln025_cad_path)
    pos_data = torch.load(cln025_pos_path)
    
    return ca_distance_data, pos_data
    
    
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
    
def kabsch_rmsd_loss(
    x0_pos: torch.Tensor,
    target_pos: torch.Tensor,
    system_id: str,
) -> torch.Tensor:
    """
    Compute the Kabsch RMSD loss between two sets of positions.
    
    """
    if system_id == "CLN025":
        alpha_carbon_idx = [4, 25, 46, 60, 72, 87, 101, 108, 122, 146]
        target_alpha_carbon_pos = target_pos[alpha_carbon_idx]
    
    else:
        raise ValueError(f"System ID {system_id} not supported")
    
    kabsch_loss = kabsch_rmsd(x0_pos, target_alpha_carbon_pos)
    
    return kabsch_loss


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

    loss = torch.tensor(0.0, device=device)
    num_graphs = len(batch)
    system_size = batch[0].pos.shape[0]
    system_id = batch[0].system_id
    for i in range(num_graphs):
        x0_pos = x0.pos[i * system_size : (i+1) * system_size]
        target_pos = target[i]
        loss = loss + kabsch_rmsd_loss(x0_pos, target_pos, system_id)
    
    loss = loss / (num_graphs * n_replications)

    return loss


def _rollout(
    batch: Batch,
    sdes: dict[str, SDE],
    score_model,
    mid_t: float,
    N_rollout: int,
    device: torch.device,
    mlcv: torch.Tensor = None,
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
    score_mid_t = get_score(batch=x_mid, sdes=sdes, t=mid_t_expanded, score_model=score_model)[
        "pos"
    ]

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
    def __init__(self, current_data_ca_distance, timelag_data_pos):
        self.current_data_ca_distance = current_data_ca_distance
        self.timelag_data_pos = timelag_data_pos

    def __len__(self):
        return len(self.current_data_ca_distance)

    def __getitem__(self, idx):
        return {
            "ca_distance": self.current_data_ca_distance[idx],
            "timelag_pos": self.timelag_data_pos[idx]
        }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = init_parser()
    args = parser.parse_args()
    sequence = args.sequence
    method = args.method
    args.date = date_str
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
    model_config["score_model"]["condition_mode"] = args.condition_type
    wandb.config.update(model_config)
    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])
    if args.score_model_mode == "train":
        score_model.train()
    else:
        score_model.eval()
    wandb.watch(score_model)
    
    # NOTE: trail, add zero conv MLP to score model
    # hidden_dim = 512
    if args.condition_type == "backbone":
        hidden_dim = 3
    elif args.condition_type == "latent" or args.condition_type == "input":
        hidden_dim = 512
    else:
        raise ValueError(f"Condition type {args.condition_type} not supported")
        
    score_model.model_nn.add_module(f"zero_conv_mlp", nn.Sequential(
        nn.Linear(hidden_dim + mlcv_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    ))
    
    # Load MLCV model
    if method == "ours" or method == "debug":
        mlcv_model = load_ours(mlcv_dim=mlcv_dim).to(device)
    elif method == "tda":
        mlcv_model = load_tda().to(device)
    elif method == "vde":
        mlcv_model = load_vde().to(device)
    else:
        raise ValueError(f"Method {method} not supported")
    print(mlcv_model)
    
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Load simulation data
    time_lag = args.time_lag
    current_data_ca_distance, timelag_data_pos = load_data()
    chemgraph = (
        get_context_chemgraph(sequence=sequence)
        .replace(system_id="CLN025")
        .to(device)
    )
    
    # Preprocess simulation data
    data_num = current_data_ca_distance.shape[0]
    train_idx = torch.from_numpy(np.random.choice(data_num - time_lag - 1, size=data_num // 100, replace=False))
    current_data_ca_distance = current_data_ca_distance[train_idx]
    current_data_ca_distance = current_data_ca_distance.to(device)
    timelag_data_pos = timelag_data_pos[train_idx + time_lag]
    timelag_data_pos = timelag_data_pos.to(device)
    dataset = CaDistanceDataset(current_data_ca_distance, timelag_data_pos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(dataloader)

    # Training loop
    total_loss = 0
    pbar = tqdm(
        range(num_epochs),
        desc=f"Loss: {total_loss:.6f}",
        total=num_epochs,
    )
    for epoch in pbar:
        total_loss = 0
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
            batch_data_ca_distance = batch["ca_distance"]
            batch_data_ca_distance_timelag = batch["timelag_pos"]
            mlcv = mlcv_model(batch_data_ca_distance)

            loss = calc_tlc_loss(
                score_model=score_model,
                mlcv=mlcv,
                target=batch_data_ca_distance_timelag,
                sdes=sdes,
                batch=[chemgraph] * batch_data_ca_distance.shape[0],
                n_replications=1,
                mid_t=mid_t,
                N_rollout=N_rollout,
                condition_mode=args.condition_type,
            )
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss
        
        scheduler.step()
        # Print epoch statistics
        pbar.set_description(f"Loss: {total_loss/num_batches:.6f}")
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "loss": total_loss/num_batches,
            "lr": current_lr,
            "epoch": epoch,
            "score_model_mode_status": score_model.training,
        })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'mlcv_state_dict': mlcv_model.state_dict(),
                'model_state_dict': score_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"model/{date_str}/checkpoint_{epoch+1}.pt")
            
    torch.save({
        'mlcv_state_dict': mlcv_model.state_dict(),
    }, f"model/{date_str}/mlcv.pt")
    
    run.finish()
    
            
if __name__ == "__main__":
    main()