import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import torch
import lightning
import os
import argparse
import pandas as pd


from matplotlib.colors import LogNorm
from tqdm import tqdm
from itertools import combinations
from scipy.stats import pearsonr


from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Transform


# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global settings
np.bool = np.bool_
CUDA_DEVICE = 0
blue = (70 / 255, 110 / 255, 250 / 255)

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

class TICA_WRAPPER:
    """TICA wrapper for coordinate transformation."""
    def __init__(self, tica_model_path, pdb_path, tica_switch: bool = False):
        with open(tica_model_path, 'rb') as f:
            self.tica_model = pickle.load(f)
        self.pdb = md.load(pdb_path)
        self.ca_resid_pair = np.array(
            [(a.index, b.index) for a, b in combinations(list(self.pdb.topology.residues), 2)]
        )
        self.tica_switch = tica_switch
        self.r_0 = 0.8
        self.nn = 6
        self.mm = 12
        print(f"Loaded TICA model: {self.tica_model}")

    def transform(self, cad_data: np.ndarray):
        # if self.tica_switch:
        #     cad_data = (1 - np.power(cad_data / self.r_0, self.nn)) / (1 - np.power(cad_data / self.r_0, self.mm))
        tica_coord = self.tica_model.transform(cad_data)
        return tica_coord

    def pos2cad(self, pos_data: np.ndarray):
        self.pdb.xyz = pos_data
        ca_pair_distances, _ = md.compute_contacts(
            self.pdb, scheme="ca", contacts=self.ca_resid_pair, periodic=False
        )
        return ca_pair_distances

def foldedness_by_hbond(
    traj: md.Trajectory,
    distance_cutoff: float = 0.35,
    bond_number_cutoff: int = 3,
):
    """
    Generate binary labels for folded/unfolded states based on hydrogen bonds.
    Only works for CLN025 molecule.
    """
    # TYR1N-YR10OT1
    donor_idx = traj.topology.select('residue 1 and name N')[0]
    acceptor_idx = traj.topology.select('residue 10 and name O')[0]
    distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
    label_TYR1N_TYR10OT1 = ((distance[:,0] < distance_cutoff)).astype(int)

    # TYR1N-YR10OT2
    donor_idx = traj.topology.select('residue 1 and name N')[0]
    acceptor_idx = traj.topology.select('residue 10 and name OXT')[0]
    distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
    label_TYR1N_TYR10OT2 = ((distance[:,0] < distance_cutoff)).astype(int)

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

    # Sum all bonds
    bond_sum = (label_TYR1N_TYR10OT1 + label_TYR1N_TYR10OT2 + label_ASP3N_TYR8O + 
                label_THR6OG1_ASP3O + label_THR6N_ASP3OD1 + label_THR6N_ASP3OD2 + 
                label_GLY7N_ASP3O + label_TYR10N_TYR1O)
    labels = bond_sum >= bond_number_cutoff

    return labels, bond_sum

def load_model_and_data(model_type, molecule, date=None):
    """Load model and data based on type."""
    simulation_idx = 0
    data_paths = {
        'pos': f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-pos.pt",
        'cad': f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-cad.pt",
        'cad-switch': f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-cad-switch.pt"
    }
    pos_torch = torch.load(data_paths['pos'])
    cad_torch = torch.load(data_paths['cad'])
    cad_switch_torch = torch.load(data_paths['cad-switch'])
    if molecule == "CLN025":
        TICA_SWITCH = True
    else:
        TICA_SWITCH = False
    
    if model_type == "mlcv":
        if molecule == "CLN025":
            INPUT_DIM = 45
            MLCV_DIM = 1
            dim_normalization = False
            date = date or "0816_171833"
        elif molecule == "2JOF":
            INPUT_DIM = 190
            MLCV_DIM = 1
            dim_normalization = False
            date = date or "0812_125552"
        
        # MLCV model configuration
        # save_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/{date}/mlcv_model.pt"
        # model_state = torch.load(save_path)
        # mlcv_state_dict = model_state["mlcv_state_dict"]
        # encoder_layers = [INPUT_DIM, 100, 100, MLCV_DIM]
        # options = {
        #     "encoder": {
        #         "activation": "tanh",
        #         "dropout": [0.1, 0.1, 0.1]
        #     },
        #     "norm_in": {},
        # }
        
        # mlcv_model = MLCV(
        #     mlcv_dim=MLCV_DIM,
        #     encoder_layers=encoder_layers,
        #     dim_normalization=dim_normalization,
        #     options=options
        # )
        # mlcv_model.load_state_dict(mlcv_state_dict)
        # mlcv_model.eval()
        save_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/{date}/mlcv_model-jit.pt"
        mlcv_model = torch.jit.load(save_path)
        mlcv_model.eval()
        
    elif model_type == "tda":
        tda_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/_baseline_/tda-{molecule}-jit.pt"
        mlcv_model = torch.jit.load(tda_path)
        mlcv_model.eval()
    
    elif model_type == "tica":
        tica_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/_baseline_/tica-{molecule}-jit.pt"
        mlcv_model = torch.jit.load(tica_path)
        mlcv_model.eval()
        
    elif model_type == "tae":
        tae_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/_baseline_/tae-{molecule}-jit.pt"
        mlcv_model = torch.jit.load(tae_path)
        mlcv_model.eval()
        
    elif model_type == "vde":
        vde_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/_baseline_/vde-{molecule}-jit.pt"
        mlcv_model = torch.jit.load(vde_path)
        mlcv_model.eval()
        
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Setup TICA wrapper
    lag = 10
    if TICA_SWITCH:
        tica_model_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}_tica_model_switch_lag{lag}.pkl"
    else:
        tica_model_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}_tica_model_lag{lag}.pkl"         
    tica_wrapper = TICA_WRAPPER(
        tica_model_path=tica_model_path,
        pdb_path=f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule}/{molecule}_from_mae.pdb",
        tica_switch=TICA_SWITCH
    )
    
    # Load committor model
    committor_path = "/home/shpark/prj-mlcv/lib/bioemu/notebook/committor.pt"
    committor_model = torch.jit.load(committor_path, map_location=f"cuda:{CUDA_DEVICE}")
    
    return mlcv_model, tica_wrapper, committor_model, pos_torch, cad_torch, cad_switch_torch

def load_reference_structure(pdb_path, tica_wrapper):
    """Load reference structure and compute its CAD representation."""
    ref_traj = md.load(pdb_path)
    ref_pos = ref_traj.xyz[0]  # Get first (and only) frame
    ref_cad = tica_wrapper.pos2cad(ref_pos.reshape(1, -1, 3))
    return ref_cad

def compute_cv_values(mlcv_model, cad_torch, model_type, reference_cad=None):
    """Compute CV values from the model with optional sign flipping."""
    cv = mlcv_model(cad_torch)
    cv = cv.detach().cpu().numpy()
    MLCV_DIM = cv.shape[1]
    
    if model_type == "mlcv":
        # Normalize CV values for MLCV
        cv_normalized = np.zeros_like(cv)
        
        for cv_dim in range(MLCV_DIM):
            cv_dim_val = cv[:, cv_dim]
            cv_range_min, cv_range_max = cv_dim_val.min(), cv_dim_val.max()
            cv_range_mean = (cv_range_min + cv_range_max) / 2.0
            cv_range = (cv_range_max - cv_range_min) / 2.0
            cv_normalized[:, cv_dim] = (cv_dim_val - cv_range_mean) / cv_range
        
        cv = cv_normalized
        
    # Additional sign flipping based on reference structure
    if reference_cad is not None:
        ref_cv = mlcv_model(torch.from_numpy(reference_cad))
        ref_cv = ref_cv.detach().cpu().numpy()
        
        # Normalize reference CV the same way
        if model_type == "mlcv":
            ref_cv_normalized = np.zeros_like(ref_cv)
            for cv_dim in range(MLCV_DIM):
                ref_cv_dim_val = ref_cv[:, cv_dim]
                cv_dim_val = cv[:, cv_dim]
                cv_range_min, cv_range_max = cv_dim_val.min(), cv_dim_val.max()
                cv_range_mean = (cv_range_min + cv_range_max) / 2.0
                cv_range = (cv_range_max - cv_range_min) / 2.0
                ref_cv_normalized[:, cv_dim] = (ref_cv_dim_val - cv_range_mean) / cv_range
            ref_cv = ref_cv_normalized
        
        # Flip signs to ensure reference CV is positive
        for cv_dim in range(MLCV_DIM):
            if ref_cv[0, cv_dim] < 0:
                cv[:, cv_dim] = -cv[:, cv_dim]
                print(f"Flipped sign for CV dimension {cv_dim} to ensure positive reference value")
    
    return cv

def check_image_exists(img_dir, filename):
    """Check if image file already exists."""
    return os.path.exists(os.path.join(img_dir, f"{filename}.png"))

def plot_tica_cv_analysis(cv, tica_data, model_type, molecule, img_dir, date=None):
    x = tica_data[:, 0]
    y = tica_data[:, 1]
    MLCV_DIM = cv.shape[1]
    os.makedirs(img_dir, exist_ok=True)
    
    for cv_dim in range(MLCV_DIM):
        # 2D TICA hexbin plot
        filename = f"tica-cv{cv_dim}_{model_type}"
        if date:
            filename += f"_{date}"
        
        if check_image_exists(img_dir, filename):
            print(f"Skipping {filename}.png - already exists")
            continue
        print(f"Plotting TICA-CV analysis for {model_type} {molecule}")
        
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        hb = ax.hexbin(
            x, y, C=cv[:, cv_dim],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
        plt.colorbar(hb)
        plt.xlabel("TIC 1")
        plt.ylabel("TIC 2")
        plt.title(f"CV {cv_dim} - {model_type.upper()}")
        plt.savefig(os.path.join(img_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3D scatter plot
        filename_3d = f"tica3d-cv{cv_dim}_{model_type}"
        if date:
            filename_3d += f"_{date}"
            
        if check_image_exists(img_dir, filename_3d):
            print(f"Skipping {filename_3d}.png - already exists")
            continue
            
        z = cv[:, cv_dim]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.8)
        ax.set_xlabel('TIC 1')
        ax.set_ylabel('TIC 2')
        ax.set_zlabel(f'CV {cv_dim}')
        ax.set_title(f'3D Scatter: CV {cv_dim} - {model_type.upper()}')
        ticks = np.arange(-1.0, 1.1, 0.5)   # [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_zticks(ticks)
        ax.view_init(azim=-85)
        
        plt.savefig(os.path.join(img_dir, f"{filename_3d}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_cv_tica_analysis(cv, tica_data, model_type, molecule, img_dir, date=None):
    """Plot 2D histogram with CV values as axes and TICA values as colors."""
    # Only works for MLCV with dimension larger than 2
    if cv.shape[1] < 2:
        return
    
    os.makedirs(img_dir, exist_ok=True)
    
    # Plot for TICA-1 as colors
    filename_tica1 = f"cv2d-tica1_{model_type}"
    if date:
        filename_tica1 += f"_{date}"
    
    if not check_image_exists(img_dir, filename_tica1):
        print(f"Plotting CV 2D histogram colored by TICA-1 for {model_type} {molecule}")
        
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        hb = ax.hexbin(
            cv[:, 0], cv[:, 1], C=tica_data[:, 0],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
        plt.colorbar(hb, label='TICA-1')
        plt.xlabel("CV 0")
        plt.ylabel("CV 1")
        plt.title(f"CV 2D Histogram (colored by TICA-1) - {model_type.upper()}")
        plt.savefig(os.path.join(img_dir, f"{filename_tica1}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Skipping {filename_tica1}.png - already exists")
    
    # Plot for TICA-2 as colors
    filename_tica2 = f"cv2d-tica2_{model_type}"
    if date:
        filename_tica2 += f"_{date}"
    
    if not check_image_exists(img_dir, filename_tica2):
        print(f"Plotting CV 2D histogram colored by TICA-2 for {model_type} {molecule}")
        
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        hb = ax.hexbin(
            cv[:, 0], cv[:, 1], C=tica_data[:, 1],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
        plt.colorbar(hb, label='TICA-2')
        plt.xlabel("CV 0")
        plt.ylabel("CV 1")
        plt.title(f"CV 2D Histogram (colored by TICA-2) - {model_type.upper()}")
        plt.savefig(os.path.join(img_dir, f"{filename_tica2}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Skipping {filename_tica2}.png - already exists")

def plot_cv_histogram(cv, model_type, molecule, img_dir, date=None):
    MLCV_DIM = cv.shape[1]
    n_bins = 50
    os.makedirs(img_dir, exist_ok=True)
    
    for cv_dim in range(MLCV_DIM):
        filename = f"cv{cv_dim}_histogram_{model_type}_{molecule}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"Skipping {filename}.png - already exists")
            continue
        print(f"Plotting CV histogram for {model_type} {molecule}")

        cv_dim_val = cv[:, cv_dim]
        
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)
        counts, bins, patches = ax.hist(
            cv_dim_val,
            bins=n_bins,
            alpha=0.7,
            color=blue,
            edgecolor='black',
            linewidth=0.5,
            log=True,
        )

        # Add statistics
        mean_val = np.mean(cv_dim_val)
        std_val = np.std(cv_dim_val)
        
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}'
        ax.text(0.75, 0.75, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')

        ax.set_xlabel(f'CV {cv_dim} Values', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Histogram of CV {cv_dim} Values - {model_type.upper()}', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_bond_analysis(cv, pos_torch, tica_wrapper, molecule, model_type, img_dir, date=None):
    if molecule != "CLN025":
        print(f"Bond analysis not available for {molecule}")
        return
    
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'brown', 'magenta']

    for cv_dim in range(MLCV_DIM):
        filename = f"bonds-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"Skipping {filename}.png - already exists")
            continue
        print(f"Plotting bond number analysis for {model_type} {molecule}")
        
        dummy_pdb = tica_wrapper.pdb
        dummy_pdb.xyz = pos_torch.numpy()
        label, bond_num = foldedness_by_hbond(dummy_pdb)
            
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        x = cv[:, cv_dim]
        y = bond_num
        grouped = [x[y == i] for i in sorted(np.unique(y))]
        violin = plt.violinplot(grouped, positions=sorted(np.unique(y)), showmeans=False, showmedians=True)
        
        for i, body in enumerate(violin['bodies']):
            body.set_facecolor(colors[i % len(colors)])
            body.set_alpha(0.7)

        violin['cbars'].set_edgecolor('gray')
        violin['cmaxes'].set_edgecolor('gray')
        violin['cmins'].set_edgecolor('gray')

        ax.set_xlabel("Bond Number")
        ax.set_ylabel("CV")
        ax.set_title(f"CV {cv_dim} vs Bond Number - {model_type.upper()}")
        
        plt.savefig(os.path.join(img_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_committor_analysis(cv, cad_torch, committor_model, model_type, molecule, img_dir, date=None):
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]

    for cv_dim in range(MLCV_DIM):
        filename = f"committor-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"Skipping {filename}.png - already exists")
            continue
        print(f"Plotting committor analysis for {model_type} {molecule}")
        committor_value = committor_model(cad_torch.to(CUDA_DEVICE))
        committor_value = committor_value.cpu().detach().numpy().flatten()
            
        # Scatter plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(committor_value, cv[:, cv_dim], color=blue, s=2)
        correlation, p_value = pearsonr(committor_value, cv[:, cv_dim])
        correlation_text = f'Pearson r = {correlation:.4f}'
        ax.text(
            0.35, 0.05,
            correlation_text,
            transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10
        )
        ax.set_xlabel("Committor")
        ax.set_ylabel(f"CV {cv_dim}")
        ax.set_title(f"CV {cv_dim} vs Committor - {model_type.upper()}")
        
        plt.savefig(os.path.join(img_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_rmsd_analysis(cv, molecule, model_type, img_dir, date=None):
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    
    for cv_dim in range(MLCV_DIM):
        filename = f"RMSD-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"Skipping {filename}.png - already exists")
            continue
        print(f"Plotting RMSD vs CV analysis for {model_type} {molecule}")
        rmsd_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-rmsd.pt"
        if not os.path.exists(rmsd_path):
            print(f"RMSD data not found at {rmsd_path}")
            return
        rmsd = torch.load(rmsd_path).numpy()
            
        # Scatter plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(rmsd, cv[:, cv_dim], color=blue, s=1, alpha=0.5)
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(rmsd, cv[:, cv_dim])
        correlation_text = f'Pearson r = {correlation:.4f}'
        ax.text(0.05, 0.05, correlation_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        ax.set_xlabel(f"RMSD to folded state, {molecule}")
        ax.set_ylabel(f"CV {cv_dim}")
        ax.set_title(f"CV {cv_dim} vs RMSD to folded state - {model_type.upper()}")
        
        plt.savefig(os.path.join(img_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def analyze_correlations(cv, committor_value, tica_data, bond_num, molecule, model_type):
    print(f"Analyzing correlations for {model_type} {molecule}")
    MLCV_DIM = cv.shape[1]
    
    print(f"\n{model_type.upper()} - {molecule} Correlation Analysis")
    print("=" * 60)

    correlation_results = []
    for cv_dim in range(MLCV_DIM):
        cv_values = cv[:, cv_dim]
        correlation, p_value = pearsonr(committor_value, cv_values)
        
        correlation_results.append({
            'CV_Dimension': cv_dim,
            'Correlation': correlation,
            'P_Value': p_value,
            'Correlation_Strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'
        })
        
        print(f"CV {cv_dim} vs Committor:")
        print(f"  Pearson correlation: {correlation:.6f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  Correlation strength: {correlation_results[-1]['Correlation_Strength']}")

    # Additional correlations
    tica_x = tica_data[:, 0]
    tica_y = tica_data[:, 1]
    
    corr_tica_x, p_tica_x = pearsonr(committor_value, tica_x)
    corr_tica_y, p_tica_y = pearsonr(committor_value, tica_y)
    
    print(f"\nCommittor vs TICA-1: r = {corr_tica_x:.6f}, p = {p_tica_x:.2e}")
    print(f"Committor vs TICA-2: r = {corr_tica_y:.6f}, p = {p_tica_y:.2e}")
    
    if bond_num is not None:
        corr_bond, p_bond = pearsonr(committor_value, bond_num)
        print(f"Committor vs Bond Number: r = {corr_bond:.6f}, p = {p_bond:.2e}")

def main():
    parser = argparse.ArgumentParser(description='Run CV analysis for different model types')
    parser.add_argument('--model_type', choices=['mlcv', 'tda', 'tica', 'tae', 'vde', 'all'], default='all',
                        help='Model type to analyze')
    parser.add_argument('--molecule', choices=['CLN025', '2JOF', '2F4K'], default='CLN025',
                        help='Molecule to analyze')
    parser.add_argument('--date', type=str, default=None,
                        help='Date string for MLCV model (only used for mlcv)')
    parser.add_argument('--img_dir', type=str, default='/home/shpark/prj-mlcv/lib/bioemu/img',
                        help='Directory to save images')
    
    args = parser.parse_args()
    
    # Create image directory if it doesn't exist
    os.makedirs(args.img_dir, exist_ok=True)
    
    model_types = ['mlcv', 'tda', 'tica', 'tae', 'vde'] if args.model_type == 'all' else [args.model_type]
    
    for model_type in model_types:
        try:
            print(f"\n{'='*60}")
            print(f"Running analysis for {model_type.upper()} - {args.molecule}")
            print(f"{'='*60}")
            
            # Load model and data
            mlcv_model, tica_wrapper, committor_model, pos_torch, cad_torch, cad_switch_torch = load_model_and_data(
                model_type, args.molecule, args.date
            )
            
            # Load reference structure
            reference_pdb_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{args.molecule}/folded.pdb"
            reference_cad = None
            if args.molecule in ["CLN025","2JOF","2F4K"] and os.path.exists(reference_pdb_path):
                reference_cad = load_reference_structure(reference_pdb_path, tica_wrapper)
                print(f"Loaded reference structure from {reference_pdb_path}")
            else:
                print(f"Reference structure not given, CV sign not aligned")
            
            # Compute CV values
            cv = compute_cv_values(mlcv_model, cad_torch, model_type, reference_cad)
            print(f"CV shape: {cv.shape}")
            print(f"CV range: {cv.max():.4f} to {cv.min():.4f}")
            
            # Compute TICA coordinates
            if args.molecule == "CLN025":
                tica_data = tica_wrapper.transform(cad_switch_torch.numpy())
            else:
                tica_data = tica_wrapper.transform(cad_torch.numpy())
            print(f"TICA shape: {tica_data.shape}")
            
            # Plot TICA-CV analysis
            plot_tica_cv_analysis(cv, tica_data, model_type, args.molecule, args.img_dir, args.date)
            
            # Plot CV-TICA analysis (2D histogram with CV as axes, TICA as colors)
            plot_cv_tica_analysis(cv, tica_data, model_type, args.molecule, args.img_dir, args.date)
            
            # Plot CV histogram
            plot_cv_histogram(cv, model_type, args.molecule, args.img_dir, args.date)
            
            # Bond analysis (only for CLN025)
            bond_num = None
            if args.molecule == "CLN025":
                plot_bond_analysis(cv, pos_torch, tica_wrapper, args.molecule, model_type, args.img_dir, args.date)
                # Get bond numbers for correlation analysis
                dummy_pdb = tica_wrapper.pdb
                dummy_pdb.xyz = pos_torch.numpy()
                _, bond_num = foldedness_by_hbond(dummy_pdb)
            
            # Committor analysis
            if args.molecule == "CLN025":
                plot_committor_analysis(cv, cad_torch, committor_model, model_type, args.molecule, args.img_dir, args.date)
            
            # RMSD analysis
            plot_rmsd_analysis(cv, args.molecule, model_type, args.img_dir, args.date)
            
            # Correlation analysis
            if args.molecule == "CLN025":
                committor_value = committor_model(cad_torch.to(CUDA_DEVICE))
                committor_value = committor_value.cpu().detach().numpy().flatten()
                analyze_correlations(cv, committor_value, tica_data, bond_num, args.molecule, model_type)
            
                print(f"\nCompleted analysis for {model_type.upper()}. Plots saved to {os.path.join(args.img_dir, args.molecule)}")
    
        except Exception as e:
            print(f"Error during analysis for {model_type.upper()}: {e}")
            continue

if __name__ == "__main__":
    main()
