# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import yaml
from torch_geometric.data.batch import Batch
from tqdm import tqdm
from itertools import combinations

import torch.nn as nn
import mdtraj as md
import pyemma


from .chemgraph import ChemGraph
from .convert_chemgraph import save_pdb_and_xtc
from .get_embeds import get_colabfold_embeds
from .model_utils import load_model, load_sdes, maybe_download_checkpoint
from .sde_lib import SDE
from .seq_io import check_protein_valid, parse_sequence, write_fasta
from .utils import (
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)
from bioemu.models import DiGConditionalScoreModel

# from mlcolvar.cvs import BaseCV, DeepTDA
# from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Transform

from model import *

logger = logging.getLogger(__name__)
HYDRA_FULL_ERROR=1

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
SupportedDenoisersLiteral = Literal["dpm", "heun"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))


@print_traceback_on_exception
@torch.no_grad()
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 10,
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1"] | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config_path: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
    method: str = "ours",
    date: str = "debug",
    mlcv_dim: int = 1,
    normalization_factor: float = 1.0,
    condition_mode: str = "none",
    ckpt_idx: str = "100",
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples, or a path to a .fasta file, or a path to an .a3m file with MSAs.
            If it is not an a3m file, then colabfold will be used to generate an MSA and embedding.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        model_name: Name of pretrained model to use. The model will be retrieved from huggingface. If not set,
           this defaults to `bioemu-v1.0`. If this is set, you do not need to provide `ckpt_path` or `model_config_path`.
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
           Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config_path` not specified. Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config_path: Path to the denoiser config, defining the denoising process.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        cache_so3_dir: Directory to store SO3 precomputations. If not set, this defaults to `~/sampling_so3_cache`.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server. If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    if ckpt_idx is not None:
        ckpt_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/{date}/checkpoint_{ckpt_idx}.pt"
    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    model_config["score_model"]["condition_mode"] = condition_mode
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    
    # NOTE: Load score model
    if condition_mode in ["input", "latent"]:
        hidden_dim = 512
    elif condition_mode in ["backbone", "backbone-both"]:
        hidden_dim = 3
    elif condition_mode in ["debug", "none"]:
        hidden_dim = 512
    else:
        raise ValueError(f"Invalid condition_mode: {condition_mode}")
    if condition_mode in ["input", "latent", "backbone"]:
        score_model.model_nn.add_module(f"zero_conv_mlp", nn.Sequential(
            nn.Linear(hidden_dim + mlcv_dim, hidden_dim),
            nn.ReLU(),
        ))
    elif condition_mode == "backbone-both":
        score_model.model_nn.add_module(f"zero_conv_mlp_pos", nn.Sequential(
            nn.Linear(hidden_dim + mlcv_dim, hidden_dim),
            nn.ReLU(),
        ))
        score_model.model_nn.add_module(f"zero_conv_mlp_orient", nn.Sequential(
            nn.Linear(2 + mlcv_dim, 3),  # 2 rotation features + mlcv_dim -> 3 axis-angle values
            nn.ReLU(),
        ))
    cond_ft_model_state = torch.load(ckpt_path, weights_only=True)
    score_model.load_state_dict(cond_ft_model_state["model_state_dict"])
    score_model.eval()
    
    # NOTE: Load MLCV model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if method == "ours":
        mlcv_model = load_ours(
            mlcv_dim=mlcv_dim,
            dim_normalization=False,
            normalization_factor=normalization_factor,
        ).to(device)
        mlcv_model.load_state_dict(cond_ft_model_state["mlcv_state_dict"])
    elif method in ["tda", "tae", "vde"]:
        mlcv_model = load_baseline(model_name=method).to(device)
    else:
        raise ValueError(f"Invalid method: {method}")
    mlcv_model.eval()
    
    cond_pdb = "/home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres.pdb"
    state_traj = md.load_pdb(cond_pdb)
    ca_resid_pair = np.array(
        [(a.index, b.index) for a, b in combinations(list(state_traj.topology.residues), 2)]
    )
    mlcv_feature, _ = md.compute_contacts(
        state_traj, scheme="ca", contacts=ca_resid_pair, periodic=False
    )
    cond_mlcv = mlcv_model(torch.from_numpy(mlcv_feature).to(device))
    print(f"MLCV feature: {mlcv_feature}")
    print(f"Conditioned MLCV: {cond_mlcv}")


    sdes = load_sdes(
        model_config_path=model_config_path,
        cache_so3_dir=cache_so3_dir
    )

    # User may have provided an MSA file instead of a sequence. This will be used for embeddings.
    msa_file = sequence if str(sequence).endswith(".a3m") else None

    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    # Parse FASTA or A3M file if sequence is a file path. Extract the actual sequence.
    # Check input sequence is valid
    sequence = parse_sequence(sequence)
    check_protein_valid(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

    if denoiser_config_path is None:
        assert (
            denoiser_type in SUPPORTED_DENOISERS
        ), f"denoiser_type must be one of {SUPPORTED_DENOISERS}"
        denoiser_config_path = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"

    with open(denoiser_config_path) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    if batch_size == 0:
        logger.warning(f"Sequence {sequence} may be too long. Attempting with batch_size = 1.")
        batch_size = 1
    logger.info(f"Using batch size {min(batch_size, num_samples)}")


    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    for seed in tqdm(
        range(existing_num_samples, num_samples, batch_size), desc="Sampling batches..."
    ):
        n = min(batch_size, num_samples - seed)
        npz_path = output_dir / format_npz_samples_filename(seed, n)
        if npz_path.exists():
            raise ValueError(
                f"Not sure why {npz_path} already exists when so far only {existing_num_samples} samples have been generated."
            )
        logger.info(f"Sampling {seed=}")
        
        cond_mlcv_expanded = cond_mlcv.repeat(min(batch_size, n), 1)
        batch = generate_batch(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=min(batch_size, n),
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
            mlcv=cond_mlcv_expanded,
            condition_mode=condition_mode,
        )
        batch = {k: v.cpu().numpy() for k, v in batch.items()}
        np.savez(npz_path, **batch, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    if set(sequences) != {sequence}:
        raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")
    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
    node_orientations = torch.tensor(
        np.concatenate([np.load(f)["node_orientations"] for f in samples_files])
    )
    backbone_pdb = "/home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres_backbone.pdb"
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        # topology_path=output_dir / "topology.pdb",
        topology_path=backbone_pdb,
        xtc_path=output_dir / "samples.xtc",
        sequence=sequence,
        filter_samples=filter_samples,
    )
    logger.info(f"Completed. Your samples are in {output_dir}.")


def get_context_chemgraph(
    sequence: str,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
) -> ChemGraph:
    n = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    single_embeds = torch.from_numpy(np.load(single_embeds_file))
    pair_embeds = torch.from_numpy(np.load(pair_embeds_file))
    assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
    assert single_embeds.shape[0] == n
    assert len(single_embeds.shape) == 2
    _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

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

    return ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
        sequence=sequence,
    )


def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    mlcv: torch.Tensor,
    cache_embeds_dir: str | Path | None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    condition_mode: str = "none",
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples, using GPU if available.

    Args:
        score_model: Score model.
        sequence: Amino acid sequence.
        sdes: SDEs defining corruption process. Keys should be 'node_orientations' and 'pos'.
        embeddings_file: Path to embeddings file.
        batch_size: Batch size.
        seed: Random seed.
        msa_file: Optional path to an MSA A3M file.
        msa_host_url: MSA server URL for colabfold.
    """

    torch.manual_seed(seed)

    context_chemgraph = get_context_chemgraph(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampled_chemgraph_batch = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
        mlcv=mlcv,
        condition_mode=condition_mode
    )
    assert isinstance(sampled_chemgraph_batch, Batch)
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")

    return {"pos": pos, "node_orientations": node_orientations}


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.DEBUG)

    fire.Fire(main)
