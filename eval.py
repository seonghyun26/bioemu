import os
import wandb
import numpy as np
import mdtraj as md
import hydra
import pyemma
import pickle
import lightning
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Statistics
from matplotlib.colors import LogNorm
from tqdm import tqdm


np.bool = np.bool_
base_dir = "/home/shpark/prj-mlcv/lib/DESRES"

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


def load_data(
    cfg : DictConfig,
) -> None:
    molecule_id = cfg.id
    simulation_id = cfg.simulation_id
    
    # Load trajectory data
    print(f"Loading trajectory data...")
    pdb_path = f"{base_dir}/DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}.pdb"
    traj_path = f"{base_dir}/DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}-{simulation_id}-protein/"
    traj_path = Path(traj_path)
    dcd_files = sorted(traj_path.glob("*.dcd"))
    traj_list = []
    for dcd_file in tqdm(
        dcd_files,
        desc="Loading trajectory data",
    ):
        traj = md.load_dcd(dcd_file, top=pdb_path)
        traj_list.append(traj)
    
    traj_all = md.join(traj_list)
    
    return traj_all
        

# Featurization and save
def featurize(
    cfg: DictConfig,
    traj_data: md.Trajectory,
) -> None:
    molecule_id = cfg.id
    simulation_id = cfg.simulation_id
    feature_type = cfg.feature
    pdb_path = f"{base_dir}/DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}.pdb"
    
    print(f"Featurizing {feature_type}...")
    state_traj = md.load(pdb_path)
    ca_atoms = state_traj.topology.select("name CA")
    n_atoms = len(ca_atoms)	
    atom_pairs = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            atom_pairs.append([ca_atoms[i], ca_atoms[j]])
    
    featurizer = pyemma.coordinates.featurizer(pdb_path)
    if 'alpha carbon distance' in feature_type:
        featurizer.add_distances(atom_pairs)
    if "dihedral angle" in feature_type:
        phi_idx, phi_value = md.compute_phi(state_traj)
        psi_idx, psi_value = md.compute_psi(state_traj)
        featurizer.add_dihedrals(phi_idx, cossin=True)
        featurizer.add_dihedrals(psi_idx, cossin=True)
    features = featurizer.transform(traj_data)
    
    tica_path = "/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-0-protein/CLN025_tica_trans.pkl"
    with open(tica_path, 'rb') as f:
        tica_model = pickle.load(f)
    tica_data = tica_model.transform(features)
    
    # Load model
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
    )
    print(f"Loading model and CVs...")
    model_input_featurizer = pyemma.coordinates.featurizer(pdb_path)
    model_input_featurizer.add_distances(atom_pairs)
    model_input = model_input_featurizer.transform(traj_data)
    model_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/{cfg.model}/mlcv.pt"
    mlcv_model.load_state_dict(torch.load(model_path)['mlcv_state_dict'])
    model_input = torch.from_numpy(model_input)
    mlcv = mlcv_model(model_input)
    mlcv = mlcv.detach().cpu().numpy()
    
    # Normalization
    mlcv_min = mlcv.min()
    mlcv_max = mlcv.max()
    mlcv_normalized = 2 * (mlcv - mlcv_min) / (mlcv_max - mlcv_min) - 1
    
    return tica_data, mlcv_normalized



# Draw TICA plot
def plot(
    cfg: DictConfig,
    tica_data: np.ndarray,
    mlcv: np.ndarray,
) -> None:
    molecule_id = cfg.id
    simulation_id = cfg.simulation_id
    save_path = f"{base_dir}/DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}_{cfg.model}_tica.png"
    
    print(f"Drawing TICA plot...")
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.hexbin(
        tica_data[:, 0],
        tica_data[:, 1],
        C=mlcv,
        gridsize=200,
        cmap='viridis',
        reduce_C_function=np.mean,
    )
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    # ax.invert_yaxis()
    plt.show()
    plt.savefig(save_path)
    plt.close()
    
    wandb.log({
        "tica": wandb.Image(save_path),
    })
    
    ran = [
        [-1, -0.5],
        [-0.5, 0],
        [0, 0.5],
        [0.5, 1],
    ]
    # Plot
    for idx, bound in enumerate(ran):
        save_path = f"{base_dir}/DESRES-Trajectory_{molecule_id}-{simulation_id}-protein/{molecule_id}_{cfg.model}_tica_{idx}.png"
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        mask  = (mlcv > bound[0]) & (mlcv < bound[1])
        mask = mask.reshape(-1)
        hb = ax.hexbin(
            tica_data[mask, 0], tica_data[mask, 1], C=mlcv[mask],  # data
            gridsize=200,                     # controls resolution
            reduce_C_function=np.mean,       # compute average per hexagon
            cmap='viridis',                  # colormap
            vmin=-1, vmax=1
        )
        plt.colorbar(hb)
        plt.xlabel("TIC 1")
        plt.ylabel("TIC 2")
        plt.savefig(save_path)
        plt.close()
        
        wandb.log({
            f"tica-{idx}": wandb.Image(save_path),
        })
    
    return 



@hydra.main(
  version_base=None,
  config_path="config",
  config_name="cln025",
)
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        project="DESRES",
        entity="eddy26",
        config=OmegaConf.to_container(cfg)
    )

    traj_data = load_data(cfg)
    tica_data, mlcv = featurize(cfg, traj_data)
    plot(cfg, tica_data, mlcv)
    
    wandb.finish()
    

if __name__ == "__main__":
    main()