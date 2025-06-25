import logging
import os
import typing
import wandb
import hydra
import numpy as np
import torch
import yaml
import lightning


from torch_geometric.data import Batch
from tqdm import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import get_score, dpm_solver
from bioemu.sde_lib import SDE
from bioemu.chemgraph import ChemGraph
from bioemu.models import EVOFORMER_EDGE_DIM, EVOFORMER_NODE_DIM
from bioemu.models import DiGConditionalScoreModel
from bioemu.structure_module import SAEncoderLayer
from bioemu import get_colabfold_embeds
from bioemu.so3_sde import SO3SDE

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Transform


# SINGLE_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_single.npy"
# PAIR_EMBED_FILE = "/home/shpark/.bioemu_embeds_cache/539b322bacb5376ca1c0a5ccad3196eb77b38dae8a09ae4a6cb83f40826936a7_pair.npy"

sequence = "GYDPETGTWG"
seed = 0

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


def load_data(batch_size=8192):
    cln025_path = "/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-0-protein/CLN025-0-CAdistance.pt"
    data = torch.load(cln025_path)
    
    time_lag = 5
    current_data = data[time_lag:]
    timelagged_data = data[:-time_lag]
    
    # Sequence encoding with embedding file
    torch.manual_seed(seed)
    n = len(sequence)
    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
    )
    single_embeds = np.load(single_embeds_file)
    pair_embeds = np.load(pair_embeds_file)
    _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]
    single_embeds, pair_embeds = torch.from_numpy(single_embeds), torch.from_numpy(pair_embeds)
    pair_embeds = pair_embeds.view(n**2, n_pair_feats)
    edge_index = torch.cat(
        [
            torch.arange(n).repeat_interleave(n).view(1, n**2),
            torch.arange(n).repeat(n).view(1, n**2),
        ],
        dim=0,
    )
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))
    chemgraph = ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
    )
    context_batch = Batch.from_data_list([chemgraph for _ in range(batch_size)])


    return current_data, timelagged_data, context_batch
    
# NOTE: Fine-tuning in ppft matter
# - Much faster
# - No need for orientation since we only need position data
def calc_tlc_loss(
    score_model: torch.nn.Module,
    sdes: dict[str, SDE],
    batch: list[ChemGraph],
    n_replications: int,
    mid_t: float,
    N_rollout: int,
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
    )

    loss = torch.tensor(0.0, device=device)
    
    # NOTE: compute eulicdean distance between x0 and time-lagged data

    return loss


def _rollout(
    batch: Batch,
    sdes: dict[str, SDE],
    score_model,
    mid_t: float,
    N_rollout: int,
    device: torch.device,
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
    

def train_with_denoising_loss(
    score_model: torch.nn.Module,
    sdes: dict[str, SDE],
    batch: list[ChemGraph],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mlcv: torch.Tensor = None,
):
    """
    Train the score model using denoising diffusion loss on custom 3D position data.
    
    Args:
        score_model: The neural network that predicts the score function
        sdes: Dictionary of SDEs for different components (positions, orientations)
        batch: List of ChemGraph objects containing your 3D position data
        optimizer: The optimizer for training
        device: The device to run the training on
    """
    # Convert list of ChemGraphs to a batch
    batch = Batch.from_data_list(batch)
    batch = batch.to(device)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Sample a random timestep t
    t = torch.rand(batch.num_graphs, device=device)
    
    # Add noise to the data according to the SDE
    pos_sde = sdes["pos"]
    batch_idx = batch.batch
    
    # Sample noisy positions
    alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch.pos, t=t, batch_idx=batch_idx)
    z = torch.randn_like(batch.pos)
    noisy_pos = alpha_t * batch.pos + sigma_t * z
    
    # Create a new batch with noisy positions
    noisy_batch = batch.clone()
    noisy_batch.pos = noisy_pos
    
    # Compute the score prediction
    score = _get_score(
        batch=noisy_batch,
        t=t,
        score_model=score_model,
        sdes=sdes,
        mlcv=mlcv,
    )
    
    # Calculate the denoising score matching loss
    # The target score is -z/sigma_t
    target_score = -z / sigma_t
    
    # MSE loss between predicted and target scores
    loss = torch.mean((score["pos"] - target_score) ** 2)
    
    # Backpropagate and update model parameters
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    run = wandb.init(
        project="bioemu-ctrl",
        entity="eddy26",
        tags=["pilot"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    ckpt_path = "./model/checkpoint.ckpt"
    cfg_path = "./model/config.yaml"
    with open(cfg_path) as f:
        model_config = yaml.safe_load(f)
    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    # score_model.load_pretrained(model_state)
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])
    
    # Load MLCV model
    encoder_layers = [45, 30, 30, 1]
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
    ).to(device)
    
    # Training loop
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)
    num_epochs = 100
    batch_size = 8192
    
    # Load your 3D position data
    train_data, time_lagged_data, context_batch = load_data(batch_size=batch_size)
    
    for epoch in range(num_epochs):
        # Create batches
        num_batches = len(train_data) // batch_size
        total_loss = 0
        
        for i in range(num_batches):
            # Get a batch of data
            batch_data = train_data[i*batch_size:(i+1)*batch_size]
            
            # Compute MLCVs for batch data
            pos = batch_data.pos
            ca_dis = pos
            mlcv = mlcv_model(ca_dis)
            
            # Train on this batch
            # loss = train_with_denoising_loss(
            #     score_model=score_model,
            #     sdes=sdes,
            #     batch=batch_data,
            #     optimizer=optimizer,
            #     device=device,
            #     mlcv=mlcv,
            # )
            loss = calc_tlc_loss(
                score_model=score_model,
                sdes=sdes,
                batch=batch_data,
                n_replications=n_replications,
                mid_t=mid_t,
                N_rollout=N_rollout,
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/num_batches:.6f}")
        
        # # Save checkpoint
        # if (epoch + 1) % 10 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': score_model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, f"checkpoint_epoch_{epoch+1}.pt")
    
    run.finish()
    
            
if __name__ == "__main__":
    main()