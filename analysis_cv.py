import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.lines
import matplotlib.patches
import matplotlib.image
import plotly.graph_objects as go
import pickle
import torch
import lightning
import os
import wandb
import argparse
import pandas as pd
import torch.nn.functional as F
import matplotlib as mpl


from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Transform

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.bool = np.bool_

def get_available_cuda_device():
    """Get the first available CUDA device, fallback to CPU if none available."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    
    # Try cuda:1 first (as used in notebook), then cuda:0
    for device_id in [1, 0]:
        try:
            device = f"cuda:{device_id}"
            # Test if device is accessible
            torch.tensor([1.0]).to(device)
            print(f"Using device: {device}")
            return device_id
        except RuntimeError as e:
            if "invalid device ordinal" in str(e):
                continue
            else:
                raise e
    
    # If neither cuda:0 nor cuda:1 work, use the default
    print("Using default CUDA device: cuda:0")
    return 0

blue = (70 / 255, 110 / 255, 250 / 255)
green = (100 / 255, 170 / 255, 120 / 255)
CUDA_DEVICE = get_available_cuda_device()
COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'brown', 'magenta']
SQUARE_FIGSIZE = (4, 4)
RECTANGLE_FIGSIZE = (5, 4)
BIG_RECTANGLE_FIGSIZE = (12, 6)
FONTSIZE = 20
FONTSIZE_SMALL = 16
LINEWIDTH = 1.5
mpl.rcParams['axes.linewidth'] = LINEWIDTH  # default is 0.8
mpl.rcParams['xtick.major.width'] = LINEWIDTH
mpl.rcParams['ytick.major.width'] = LINEWIDTH
mpl.rcParams['xtick.minor.width'] = LINEWIDTH
mpl.rcParams['ytick.minor.width'] = LINEWIDTH



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


def format_plot_axes(
    ax,
    fig=None,
    hide_ticks=False,
    hide_x_ticks=False,
    hide_y_ticks=False,
    show_grid=True,
    grid_alpha=0.3,
    set_axis_below=True,
    align_ylabels=False,
    model_type=None,
    show_y_labels=True,
    fontsize=FONTSIZE_SMALL,
    linewidth=LINEWIDTH,
):
    """
    Apply consistent formatting to plot axes.
    
    Args:
        ax: matplotlib axes object
        fig: matplotlib figure object (required for align_ylabels)
        hide_ticks: Hide both x and y ticks
        hide_x_ticks: Hide only x ticks
        hide_y_ticks: Hide only y ticks
        show_grid: Show grid lines
        grid_alpha: Grid transparency
        set_axis_below: Set grid behind plot elements
        align_ylabels: Align y-axis labels (requires fig)
        fontsize: Font size for tick labels
        model_type: Model type for conditional formatting
        show_y_labels: Whether to show y-axis labels (used for conditional formatting)
    """
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_linewidth(LINEWIDTH)
    # ax.spines['bottom'].set_linewidth(LINEWIDTH)
    
    # Handle tick visibility
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if hide_x_ticks:
            ax.set_xticks([])
        if hide_y_ticks:
            ax.set_yticks([])
    
    # Grid formatting
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linewidth=linewidth)
        if set_axis_below:
            ax.set_axisbelow(True)
    
    # Tick parameters
    if not hide_ticks and not (hide_x_ticks and hide_y_ticks):
        if model_type == "tda" and show_y_labels:
            # Show both x and y tick labels for TDA model
            ax.tick_params(axis='both', labelsize=fontsize)
        else:
            # Hide y-axis labels for non-TDA models unless explicitly requested
            if show_y_labels:
                ax.tick_params(axis='both', labelsize=fontsize)
            else:
                ax.tick_params(axis='both', labelsize=fontsize, labelleft=False)
    
    # Align y-labels if requested and figure is provided
    if align_ylabels and fig is not None:
        fig.align_ylabels(ax)


def load_model_and_data(
    model_type,
    molecule,
    date=None,
):
    """Load models and return data paths instead of loading large tensors into memory/GPU."""
    simulation_idx = 0
    data_paths = {
        'pos': f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-pos.pt",
        'cad': f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-cad.pt",
        'cad-switch': f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-cad-switch.pt"
    }
    # Do not load big tensors here; return paths for batch-wise processing later
    pos_path = data_paths['pos'] if molecule == "CLN025" else None
    cad_switch_path = data_paths['cad-switch'] if molecule == "CLN025" else None
    cad_path = data_paths['cad']
    if molecule == "CLN025":
        TICA_SWITCH = True
    else:
        TICA_SWITCH = False
    
    model_root_path = "/home/shpark/prj-mlcv/lib/bioemu/opes/model/_baseline_"
    if model_type == "ours":
        if molecule == "CLN025":
            date = date or "0816_171833"
        
        elif molecule == "2JOF":
            date = date or "0814_073849"
        
        elif molecule == "2F4K":
            date = date or "0819_173704"
            
        elif molecule == "1FME":
            date = date or "0904_160804"
        
        elif molecule == "GTT":
            date = date or "0905_160702"
            
        elif molecule == "NTL9":
            date = date or "0905_054344"
        
        else:
            raise ValueError(f"Invalid molecule: {molecule} for {model_type}")
        
        device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
        save_path = f"/home/shpark/prj-mlcv/lib/bioemu/opes/model/{date}-{molecule.upper()}-jit.pt"
        mlcv_model = torch.jit.load(save_path, map_location=device_str)
        mlcv_model.eval()
    
    # elif model_type == "mlcv-trans":
    #     date = date or "0825_072649"
    #     device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
    #     save_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/{date}/mlcv_model-jit.pt"
    #     mlcv_model = torch.jit.load(save_path, map_location=device_str)
    #     mlcv_model.eval()
    
    elif model_type == "tda":
        tda_path = f"{model_root_path}/tda-{molecule}-jit.pt"
        device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
        mlcv_model = torch.jit.load(tda_path, map_location=device_str)
        mlcv_model.eval()
    
    elif model_type == "tica":
        tica_path = f"{model_root_path}/tica-{molecule}-jit.pt"
        device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
        mlcv_model = torch.jit.load(tica_path, map_location=device_str)
        mlcv_model.eval()
        
    elif model_type == "tae":
        tae_path = f"{model_root_path}/tae-{molecule}-jit.pt"
        device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
        mlcv_model = torch.jit.load(tae_path, map_location=device_str)
        mlcv_model.eval()
        
    elif model_type == "vde":
        vde_path = f"{model_root_path}/vde-{molecule}-jit.pt"
        device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
        mlcv_model = torch.jit.load(vde_path, map_location=device_str)
        mlcv_model.eval()
        
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Setup TICA wrapper
    lag = 10
    if TICA_SWITCH:
        tica_model_path = f"./opes/data/{molecule}/{molecule}_tica_model_switch_lag{lag}.pkl"
    else:
        tica_model_path = f"./opes/data/{molecule}/{molecule}_tica_model_lag{lag}.pkl"         
    tica_wrapper = TICA_WRAPPER(
        tica_model_path=tica_model_path,
        pdb_path=f"./opes/data/{molecule}/{molecule}_from_mae.pdb",
        tica_switch=TICA_SWITCH
    )
    
    # Load committor model
    if molecule == "CLN025":
        committor_path = "./opes/data/CLN025/committor.pt"
        committor_model = torch.jit.load(committor_path, map_location=f"cuda:{CUDA_DEVICE}")
    else:
        committor_model = None
    
    # Return models and data paths for batch-wise processing
    return mlcv_model, tica_wrapper, committor_model, pos_path, cad_path, cad_switch_path

def load_reference_structure(
    pdb_path,
    tica_wrapper,
):
    """Load reference structure and compute its CAD representation."""
    ref_traj = md.load(pdb_path)
    ref_pos = ref_traj.xyz[0]  # Get first (and only) frame
    ref_cad = tica_wrapper.pos2cad(ref_pos.reshape(1, -1, 3))
    return ref_cad



def compute_cv_values(
    mlcv_model,
    cad_torch_path,
    model_type,
    molecule,
    reference_cad=None,
    batch_size=10000,
    device=None,
    cache=True,
):
    """Compute CV values from the model with optional sign flipping using batch processing."""
    cv_data_path = f"/home/shpark/prj-mlcv/lib/bioemu/opes/data/{molecule.upper()}/{model_type}_mlcv.npy"
    if os.path.exists(cv_data_path):
        print(f"> Using cached CV values from {cv_data_path}")
        cv = np.load(cv_data_path)
    
    else:
        cad_torch = torch.load(cad_torch_path).to(device)
        dataset = TensorDataset(cad_torch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Computing CV values in batches of {batch_size}...")
        
        with torch.no_grad():
            sample_batch = next(iter(dataloader))[0]
            sample_output = mlcv_model(sample_batch)
            output_dim = sample_output.shape[1]
        cv_batches = torch.zeros((len(cad_torch), output_dim)).to(cad_torch.device)
        
        with torch.no_grad():
            for batch_idx, (batch_data,) in enumerate(tqdm(
                dataloader,
                desc="Computing CV values",
                total=len(dataloader),
                leave=False,
            )):
                batch_cv = mlcv_model(batch_data)
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_cv.shape[0]  # Handle last batch size correctly
                cv_batches[start_idx:end_idx] = batch_cv
        
        cv = cv_batches.detach().cpu().numpy()
        MLCV_DIM = cv.shape[1]
        
        print(f"CV computation complete. Shape: {cv.shape}")
        
        if model_type == "ours":
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
            with torch.no_grad():
                ref_cv = mlcv_model(torch.from_numpy(reference_cad).to(cad_torch.device))
                ref_cv = ref_cv.detach().cpu().numpy()
            
            # Normalize reference CV the same way
            if model_type == "ours":
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

        np.save(cv_data_path, cv)
        print(f"Saved CV values to {cv_data_path}")
        
    return cv


def foldedness_by_hbond_distance(
    traj,
    distance_cutoff: float = 0.35,
    bond_number_cutoff: int = 3
):
    """
    Optimized distance-only version of hydrogen bond labeling.
    
    Much faster than the angle-based version when angles are not needed.
    
    Args:
        traj (mdtraj): mdtraj trajectory object
        distance_cutoff (float): donor-acceptor distance cutoff in nm (default 0.35 nm = 3.5 angstrom)
        bond_number_cutoff (int): minimum number of bonds to be considered as folded (default 3)

    Returns:
        labels (np.array): binary array (1: folded, 0: unfolded)
        bond_sum (np.array): number of hydrogen bonds per frame
    """
    
    # Pre-compute all atom indices
    atom_indices = {
        'TYR1_N': traj.topology.select('residue 1 and name N')[0],
        'TYR1_O': traj.topology.select('residue 1 and name O')[0],
        'ASP3_N': traj.topology.select('residue 3 and name N')[0],
        'ASP3_O': traj.topology.select('residue 3 and name O')[0],
        'ASP3_OD1': traj.topology.select('residue 3 and name OD1')[0],
        'ASP3_OD2': traj.topology.select('residue 3 and name OD2')[0],
        'THR6_N': traj.topology.select('residue 6 and name N')[0],
        'THR6_OG1': traj.topology.select('residue 6 and name OG1')[0],
        'GLY7_N': traj.topology.select('residue 7 and name N')[0],
        'TYR8_O': traj.topology.select('residue 8 and name O')[0],
        'TYR10_N': traj.topology.select('residue 10 and name N')[0],
        'TYR10_OT1': traj.topology.select('residue 10 and name O')[0],
        'TYR10_OXT': traj.topology.select('residue 10 and name OXT')[0],
    }
    
    # Define all bond pairs
    distance_pairs = [
        [atom_indices['TYR1_N'], atom_indices['TYR10_OT1']],      # TYR1N-TYR10OT1
        [atom_indices['TYR1_N'], atom_indices['TYR10_OXT']],    # TYR1N-TYR10OT2  
        [atom_indices['ASP3_N'], atom_indices['TYR8_O']],       # ASP3N-TYR8O
        [atom_indices['THR6_OG1'], atom_indices['ASP3_O']],     # THR6OG1-ASP3O
        [atom_indices['THR6_N'], atom_indices['ASP3_OD1']],     # THR6N-ASP3OD1
        [atom_indices['THR6_N'], atom_indices['ASP3_OD2']],     # THR6N-ASP3OD2
        [atom_indices['GLY7_N'], atom_indices['ASP3_O']],       # GLY7N-ASP3O
        [atom_indices['TYR10_N'], atom_indices['TYR1_O']],      # TYR10N-TYR1O
    ]
    
    # Batch compute all distances at once
    all_distances = md.compute_distances(traj, distance_pairs)
    bond_labels = (all_distances < distance_cutoff).astype(int)
    bond_sum = np.sum(bond_labels, axis=1)
    labels = bond_sum >= bond_number_cutoff
    
    return labels, bond_sum

def get_dssp_simplified_mapping():
    """Get mapping from DSSP full codes to simplified categories."""
    return {
        'H': 'H',  # Alpha helix -> Helix
        'G': 'H',  # 3-10 helix -> Helix  
        'I': 'H',  # Pi helix -> Helix
        'E': 'E',  # Extended strand -> Sheet
        'B': 'E',  # Beta bridge -> Sheet
        'T': 'C',  # Turn -> Coil
        'S': 'C',  # Bend -> Coil
        ' ': 'C',  # Coil -> Coil
    }

def get_molecule_residue_range(
    molecule,
):
    """
    Get the residue indices and 0-indexed arrays for different molecules.
    
    Args:
        molecule: Molecule name (e.g., "CLN025", "2JOF")
        
    Returns:
        tuple: (residue_indices, residue_indices_0) where residue_indices are 1-indexed
               and residue_indices_0 are 0-indexed for array slicing
    """
    # if molecule == "CLN025":
    #     residue_indices = list(range(1, 11))  # 1-10
    #     residue_indices_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # elif molecule == "2JOF":
    #     residue_indices = list(range(1, 16))  # 1-15
    #     residue_indices_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    if molecule == "CLN025":
        residue_indices = [1, 2, 7, 8]
        residue_indices_0 = [0, 1, 6, 7]
    elif molecule == "2JOF":
        residue_indices = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
        residue_indices_0 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    elif molecule == "1FME":
        residue_indices = [4, 5, 6, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        residue_indices_0 = [3, 4, 5, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    elif molecule == "GTT":
        residue_indices = [7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 27, 28]
        residue_indices_0 = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 26, 27]
    else:
        print(f"No specific residue range defined for molecule {molecule}, using all residues")
        residue_indices = None
        residue_indices_0 = None
    
    return residue_indices, residue_indices_0

def load_and_filter_dssp_data(
    molecule,
):
    """
    Load and filter DSSP data for a specific molecule based on predefined residue ranges.
    
    Args:
        molecule: Molecule name (e.g., "CLN025", "2JOF")
        
    Returns:
        dict: Dictionary containing filtered DSSP data and metadata:
              - 'dssp_full': Filtered full DSSP data (or None if not found)
              - 'dssp_simplified': Filtered simplified DSSP data (or None if not found)
              - 'residue_indices': 1-indexed residue numbers for labels
              - 'filtered': Boolean indicating if filtering was applied
    """
    # Get residue ranges for this molecule
    residue_indices, residue_indices_0 = get_molecule_residue_range(molecule)
    
    # Load DSSP data
    dssp_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-dssp.npy"
    dssp_simplified_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-dssp-simplified.npy"
    
    # Initialize results
    result = {
        'dssp_full': None,
        'dssp_simplified': None,
        'residue_indices': residue_indices,
        'filtered': residue_indices is not None
    }
    
    # Load full DSSP data
    if os.path.exists(dssp_path):
        dssp_full = np.load(dssp_path)
        if residue_indices_0 is not None:
            result['dssp_full'] = dssp_full[:, residue_indices_0]
            print(f"Loaded and filtered DSSP full data for {molecule}: using residues {residue_indices}")
        else:
            result['dssp_full'] = dssp_full
            print(f"Loaded DSSP full data for {molecule}: using all residues")
    else:
        print(f"DSSP full data not found at {dssp_path}")
    
    # Load simplified DSSP data
    if os.path.exists(dssp_simplified_path):
        dssp_simplified = np.load(dssp_simplified_path)
        if residue_indices_0 is not None:
            result['dssp_simplified'] = dssp_simplified[:, residue_indices_0]
            print(f"Loaded and filtered DSSP simplified data for {molecule}")
        else:
            result['dssp_simplified'] = dssp_simplified
            print(f"Loaded DSSP simplified data for {molecule}: using all residues")
    else:
        print(f"DSSP simplified data not found at {dssp_simplified_path}")
    
    return result

def rasterize_plot_elements():
    """
    Apply rasterization to data elements in the current figure while keeping text/labels as vectors.
    This reduces file size for plots with many data points.
    """
    fig = plt.gcf()
    for ax in fig.get_axes():
        for child in ax.get_children():
            if hasattr(child, 'set_rasterized'):
                # Rasterize data elements but keep text/labels as vectors
                if any(isinstance(child, cls) for cls in [
                    matplotlib.collections.Collection,    # Scatter plots, hexbin, etc.
                    matplotlib.lines.Line2D,              # Line plots
                    matplotlib.patches.Rectangle,         # Bar plots, histograms
                    matplotlib.patches.Polygon,           # Violin plots
                    matplotlib.image.AxesImage,           # Heatmaps, images
                ]):
                    child.set_rasterized(True)

def check_image_exists(
    img_dir,
    filename,
):
    """Check if image file already exists."""
    png_path = os.path.join(img_dir, f"{filename}.png")
    pdf_path = os.path.join(img_dir, f"pdf/{filename}.pdf")
    return os.path.exists(png_path) and os.path.exists(pdf_path)

def format_violin_parts(
    violin_parts
):
    """Format violin plot parts with consistent styling."""
    # Customize violin plot bodies
    for i, body in enumerate(violin_parts['bodies']):
        body.set_facecolor(COLORS[i % len(COLORS)])
        body.set_alpha(0.7)
    
    # Customize violin plot lines
    violin_parts['cbars'].set_edgecolor('gray')
    violin_parts['cbars'].set_linewidth(2)
    violin_parts['cbars'].set_alpha(1)
    violin_parts['cmaxes'].set_edgecolor('gray')
    violin_parts['cmaxes'].set_linewidth(2)
    violin_parts['cmaxes'].set_alpha(1)
    violin_parts['cmins'].set_edgecolor('gray')
    violin_parts['cmins'].set_linewidth(2)
    violin_parts['cmins'].set_alpha(1)
    violin_parts['cmeans'].set_edgecolor('black')
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmeans'].set_alpha(0.5)
    # violin_parts['cmedians'].set_linewidth(0)

def save_plot_dual_format(
    img_dir,
    filename,
    dpi=200,
    bbox_inches='tight',
    pad_inches=0.1,
    rasterized=True,
    file_log_name = None,
):
    """
    Save plot in both PNG and PDF formats with existence checking.
    
    Args:
        img_dir: Directory to save images
        filename: Base filename without extension
        dpi: Resolution for PNG format
        bbox_inches: Bounding box setting for tight layout
        pad_inches: Padding for tight layout
        rasterized: Whether to rasterize the plot elements (reduces file size)
        file_log_name: Name to log to wandb
    Returns:
        bool: True if files were saved, False if they already existed
    """
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, "pdf"), exist_ok=True)
    
    png_path = os.path.join(img_dir, f"{filename}.png")
    pdf_path = os.path.join(img_dir, f"pdf/{filename}.pdf")
    
    
    # Check if both files already exist
    if check_image_exists(img_dir, filename):
        print(f"> Skipping {filename} - both PNG and PDF already exist")
        return False
    
    # Apply rasterization to plot elements if requested
    
    # Save in both formats
    try:
        # Save as PNG
        if not os.path.exists(png_path):
            plt.savefig(
                png_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
            print(f">> Saved {png_path}")
        if file_log_name == None:
            file_log_name = filename
        wandb.log({
            f"{file_log_name}": wandb.Image(str(png_path))
        })
        
        # Save as PDF
        if not os.path.exists(pdf_path):
            if rasterized:
                rasterize_plot_elements()
            plt.savefig(
                pdf_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
            print(f">> Saved {pdf_path}")
        wandb.save(pdf_path)

        return True
        
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False



# Plots
def plot_tica_cv_analysis(
    cv,
    tica_data,
    model_type,
    molecule,
    img_dir,
    date=None,
    plot_3d=False,
):
    x = tica_data[:, 0]
    y = tica_data[:, 1]
    MLCV_DIM = cv.shape[1]
    os.makedirs(img_dir, exist_ok=True)
    
    for cv_dim in range(MLCV_DIM):
        # 2D TICA hexbin plot
        filename = f"tica-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            wandb.log({
                filename: wandb.Image(str(f"{img_dir}/{filename}.png"))
            })
        else:
            print(f"> Plotting TICA-CV analysis for {model_type} {molecule}")
            fig = plt.figure(figsize=SQUARE_FIGSIZE)
            ax = fig.add_subplot(111)
            hb = ax.hexbin(
                x, y, C=cv[:, cv_dim],
                gridsize=200,
                reduce_C_function=np.mean,
                cmap='viridis',
                zorder=2,
                rasterized=True,
            )
            # plt.colorbar(hb)
            ax.set_xlabel("TIC 1", fontsize=FONTSIZE_SMALL)
            if model_type == "tda":
                ax.set_ylabel("TIC 2", fontsize=FONTSIZE_SMALL)
            # if molecule == "CLN025":
            #     ax.set_xticks([-2, -1, 0])
            #     if model_type == "tda":
            #         ax.set_yticks([-6, -4, -2, 0, 2])
            
            # Apply consistent formatting
            format_plot_axes(
                ax, fig=fig, 
                model_type=model_type, 
                show_y_labels=(model_type == "tda"),
                align_ylabels=True
            )
            save_plot_dual_format(
                img_dir, filename,
                dpi=300, bbox_inches='tight',
                file_log_name="TICA-CV"
            )
            plt.close()

        # 3D scatter plot
        filename_3d = f"tica3d-cv{cv_dim}-{model_type}"
        if date:
            filename_3d += f"_{date}"
        if check_image_exists(img_dir, filename_3d) or not plot_3d:
            print(f"> Skipping {filename_3d}.png - already exists")
            wandb.log({
                filename_3d: wandb.Image(str(f"{img_dir}/{filename_3d}.png"))
            })
        else:
            print(f"> Plotting TICA-CV 3D scatter plot for {model_type} {molecule}")
            z = cv[:, cv_dim]
            fig = plt.figure(figsize=(5, 5), layout='constrained')
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.6)
            ax.set_xlabel('TIC 1', fontsize=FONTSIZE_SMALL, labelpad=10)
            ax.set_ylabel('TIC 2', fontsize=FONTSIZE_SMALL, labelpad=10)
            ax.set_zticks([-1.0, 0.0, 1.0])

            if model_type == "ours":
                ax.set_zlabel("CV", fontsize=FONTSIZE_SMALL, rotation=90)
                ax.tick_params(axis='z', labelsize=FONTSIZE_SMALL)
            else:
                ax.tick_params(axis='z', labelsize=FONTSIZE_SMALL, labelleft=False)
            ax.set_zlim(-1.0, 1.0)
            ax.view_init(azim=-80, elev=25)
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.line.set_linewidth(LINEWIDTH)
            ax.tick_params(axis='x', labelsize=FONTSIZE_SMALL)
            ax.tick_params(axis='y', labelsize=FONTSIZE_SMALL)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.grid(True, alpha=0.3, linewidth=LINEWIDTH)

            save_plot_dual_format(
                img_dir, filename_3d,
                dpi=300, bbox_inches='tight',
                file_log_name="TICA-CV-3D"
            )
            plt.close()

def plot_cv_tica_analysis(
    cv,
    tica_data,
    model_type,
    molecule,
    img_dir,
    date=None,
):
    """Plot 2D histogram with CV values as axes and TICA values as colors."""
    # Only works for MLCV with dimension larger than 2
    if cv.shape[1] < 2:
        return
    
    os.makedirs(img_dir, exist_ok=True)
    
    # Plot for TICA-1 as colors
    filename_tica1 = f"cv2d-tica1-{model_type}"
    if date:
        filename_tica1 += f"_{date}"
    
    if not check_image_exists(img_dir, filename_tica1):
        print(f"> Plotting CV 2D histogram colored by TICA-1 for {model_type} {molecule}")
        
        fig = plt.figure(figsize=SQUARE_FIGSIZE)
        ax = fig.add_subplot(111)
        hb = ax.hexbin(
            cv[:, 0], cv[:, 1], C=tica_data[:, 0],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
        # Apply consistent formatting
        format_plot_axes(
            ax, fig=fig, 
            hide_ticks=True,
            model_type=model_type, 
            show_y_labels=False
        )
        # plt.colorbar(hb, label='TICA-1')
        # plt.xlabel("CV 0")
        # plt.ylabel("CV 1")
        # plt.title(f"CV 2D Histogram (colored by TICA-1) - {model_type.upper()}")
        save_plot_dual_format(img_dir, filename_tica1, dpi=300, bbox_inches='tight', file_log_name="CV-TICA-1")
        plt.close()
    else:
        print(f"> Skipping {filename_tica1}.png - already exists")
    
    # Plot for TICA-2 as colors
    filename_tica2 = f"cv2d-tica2-{model_type}"
    if date:
        filename_tica2 += f"_{date}"
    
    if not check_image_exists(img_dir, filename_tica2):
        print(f"> Plotting CV 2D histogram colored by TICA-2 for {model_type} {molecule}")
        
        fig = plt.figure(figsize=SQUARE_FIGSIZE)
        ax = fig.add_subplot(111)
        hb = ax.hexbin(
            cv[:, 0], cv[:, 1], C=tica_data[:, 1],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
        # Apply consistent formatting
        format_plot_axes(
            ax, fig=fig, 
            hide_ticks=True,
            model_type=model_type, 
            show_y_labels=False
        )
        # plt.colorbar(hb, label='TICA-2')
        # plt.xlabel("CV 0")
        # plt.ylabel("CV 1")
        # plt.title(f"CV 2D Histogram (colored by TICA-2) - {model_type.upper()}")
        save_plot_dual_format(img_dir, filename_tica2, dpi=300, bbox_inches='tight', file_log_name="CV-TICA-2")
        plt.close()
    else:
        print(f"> Skipping {filename_tica2}.png - already exists")

def plot_cv_histogram(
    cv,
    model_type,
    molecule,
    img_dir,
    date=None,
):
    MLCV_DIM = cv.shape[1]
    n_bins = 50
    os.makedirs(img_dir, exist_ok=True)
    
    for cv_dim in range(MLCV_DIM):
        filename = f"cv{cv_dim}_histogram-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
        print(f"> Plotting CV histogram for {model_type} {molecule}")

        cv_dim_val = cv[:, cv_dim]
        
        fig = plt.figure(figsize=RECTANGLE_FIGSIZE)
        ax = fig.add_subplot(111)
        counts, bins, patches = ax.hist(
            cv_dim_val,
            bins=n_bins,
            alpha=0.7,
            color=blue,
            edgecolor='black',
            linewidth=0.5,
            log=True,
            rasterized=True,
        )

        # Add statistics
        mean_val = np.mean(cv_dim_val)
        std_val = np.std(cv_dim_val)
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}'
        ax.text(0.75, 0.75, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Apply consistent formatting
        format_plot_axes(
            ax, fig=fig, 
            hide_ticks=True,
            model_type=model_type, 
            show_y_labels=False,
            align_ylabels=True
        )
        # ax.set_xlabel(f'CV {cv_dim} Values', fontsize=12)
        # ax.set_ylabel('Frequency', fontsize=12)
        # ax.set_title(f'Histogram of CV {cv_dim} Values - {model_type.upper()}', fontsize=14)
        save_plot_dual_format(
            img_dir, filename,
            dpi=300, bbox_inches='tight',
            file_log_name=f"CV-Histogram-{cv_dim}"
        )
        plt.close()

def plot_bond_analysis(
    cv,
    pos_torch,
    tica_wrapper,
    molecule,
    model_type,
    img_dir,
    date=None,
):
    if molecule != "CLN025":
        print(f"Bond analysis not available for {molecule}")
        return
    
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]

    for cv_dim in range(MLCV_DIM):
        filename = f"bonds-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
        print(f"> Plotting bond number analysis for {model_type} {molecule}")
        
        bond_path = "/home/shpark/prj-mlcv/lib/bioemu/opes/data/CLN025/bond_num.npy"
        if os.path.exists(bond_path):
            print(f"> Loaded bond number from {bond_path}")
            bond_num = np.load(bond_path)
        else:
            print(f"> No bond number found at {bond_path}, computing...")
            dummy_pdb = tica_wrapper.pdb
            dummy_pdb.xyz = pos_torch.cpu().detach().numpy()
            label, bond_num = foldedness_by_hbond_distance(dummy_pdb)
            np.save(bond_path, bond_num)
            
        fig = plt.figure(figsize=RECTANGLE_FIGSIZE)
        ax = fig.add_subplot(111)
        x = cv[:, cv_dim]
        y = bond_num
        grouped = [x[y == i] for i in sorted(np.unique(y))]
        violin_parts = plt.violinplot(
            grouped, positions=sorted(np.unique(y)),
            showmeans=True, showmedians=False, showextrema=True,
        )
        format_violin_parts(violin_parts)

        ax.set_xlabel("Bond Number", fontsize=FONTSIZE_SMALL)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        if model_type == "tda":
            ax.set_ylabel("CV", fontsize=FONTSIZE_SMALL)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        # Apply consistent formatting
        format_plot_axes(
            ax, fig=fig, 
            model_type=model_type, 
            show_y_labels=(model_type == "tda"),
            align_ylabels=True
        )
        # ax.set_title(f"CV {cv_dim} vs Bond Number - {model_type.upper()}")
        save_plot_dual_format(
            img_dir, filename,
            dpi=300, bbox_inches='tight',
            file_log_name=f"CV-Bond-Number-{cv_dim}"
        )
        plt.close()

def plot_committor_analysis(
    cv,
    cad_torch,
    committor_model,
    model_type,
    molecule,
    img_dir,
    date=None,
):
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]

    for cv_dim in range(MLCV_DIM):
        filename = f"committor-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
        print(f"> Plotting committor analysis for {model_type} {molecule}")
        committor_value_path = f"/home/shpark/prj-mlcv/lib/bioemu/opes/data/CLN025/committor_value_cv{cv_dim}.npy"
        if os.path.exists(committor_value_path):
            print(f"> Loaded committor value from {committor_value_path}")
            committor_value = np.load(committor_value_path)
        else:
            print(f"> No committor value found at {committor_value_path}, computing...")
            committor_value = committor_model(cad_torch.to(CUDA_DEVICE))
            committor_value = committor_value.cpu().detach().numpy().flatten()
            np.save(committor_value_path, committor_value)
            
        # Scatter plot
        fig = plt.figure(figsize=RECTANGLE_FIGSIZE)
        ax = fig.add_subplot(111)
        ax.scatter(
            committor_value, cv[:, cv_dim],
            color=blue, s=2,
            zorder=2,
            rasterized=True,
        )
        correlation_pearson, p_value_pearson = pearsonr(committor_value, cv[:, cv_dim])
        correlation_spearman, p_value_spearman = spearmanr(committor_value, cv[:, cv_dim])
        wandb.log({
            f"committor/pearson_{cv_dim}": correlation_pearson,
            f"committor/spearman_{cv_dim}": correlation_spearman,
        })
        # correlation_text = f'Pearson r = {correlation_pearson:.4f}\nSpearman ρ = {correlation_spearman:.4f}'
        # ax.text(
        #     1.05, 0.5,
        #     correlation_text,
        #     transform=ax.transAxes, 
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        #     verticalalignment='top', fontsize=10
        # )
        # ax.set_title(f"CV {cv_dim} vs Committor - {model_type.upper()}")
        ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_xlabel("Committor", fontsize=FONTSIZE_SMALL)
        if model_type == "tda":
            ax.set_ylabel(f"CV {cv_dim}", fontsize=FONTSIZE_SMALL)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        # Apply consistent formatting
        format_plot_axes(
            ax, fig=fig, 
            model_type=model_type, 
            show_y_labels=(model_type == "tda"),
            align_ylabels=True
        )

        save_plot_dual_format(
            img_dir, filename,
            dpi=300, bbox_inches='tight',
            file_log_name=f"Committor-{cv_dim}"
        )
        plt.close()

def plot_rmsd_analysis(
    cv,
    molecule,
    model_type,
    img_dir,
    date=None,
    unfolded_flag=False,
):
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    
    for cv_dim in range(MLCV_DIM):
        filename = f"RMSD-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
        print(f"> Plotting RMSD vs CV analysis for {model_type} {molecule}")
        rmsd_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-rmsd.pt"
        if not os.path.exists(rmsd_path):
            print(f"RMSD data not found at {rmsd_path}")
            return
        rmsd = torch.load(rmsd_path).numpy()
        if unfolded_flag:
            rmsd_unfolded_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-rmsd_unfolded.pt"
            if not os.path.exists(rmsd_unfolded_path):
                print(f"RMSD unfolded data not found at {rmsd_unfolded_path}")
                return
            rmsd_unfolded = torch.load(rmsd_unfolded_path).numpy()
        
        # Scatter plot
        fig = plt.figure(figsize=RECTANGLE_FIGSIZE)
        ax = fig.add_subplot(111)
        ax.scatter(
            rmsd, cv[:, cv_dim],
            color=blue, s=0.5, alpha=0.4, zorder=1,
            rasterized=True,
        )
        if unfolded_flag:
            ax.scatter(
                rmsd_unfolded, cv[:, cv_dim],
                color=green, s=0.5, alpha=0.4, zorder=1,
                rasterized=True,
            )
        
        # Calculate Pearson and Spearman correlations
        correlation_folded_p, p_value_folded_p = pearsonr(rmsd, cv[:, cv_dim])
        correlation_folded_s, p_value_folded_s = spearmanr(rmsd, cv[:, cv_dim])
        if unfolded_flag:
            correlation_unfolded_p, p_value_unfolded_p = pearsonr(rmsd_unfolded, cv[:, cv_dim])
            correlation_unfolded_s, p_value_unfolded_s = spearmanr(rmsd_unfolded, cv[:, cv_dim])
        correlation_text = (
            f'<RMSD>\n'
            f'Pearson r = {correlation_folded_p:.4f}\n'
            f'Spearman ρ = {correlation_folded_s:.4f}\n'
        )
        wandb.log({
            f"rmsd/folded/pearson_{cv_dim}": correlation_folded_p,
            f"rmsd/folded/spearman_{cv_dim}": correlation_folded_s,
        })
        if unfolded_flag:
            correlation_text += (
                f'<Unfolded>\n'
                f'Pearson r = {correlation_unfolded_p:.4f}\n'
                f'Spearman ρ = {correlation_unfolded_s:.4f}'
            )
            wandb.log({
                f"rmsd/unfolded/pearson_{cv_dim}": correlation_unfolded_p,
                f"rmsd/unfolded/spearman_{cv_dim}": correlation_unfolded_s,
            })
        # ax.text(
        #     1.05, 0.5, correlation_text, transform=ax.transAxes, 
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        #     verticalalignment='top', fontsize=10
        # )
        # ax.set_title(f"CV {cv_dim} vs RMSD to folded state - {model_type.upper()}")
        ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_xlabel(f"RMSD", fontsize=FONTSIZE_SMALL)
        if model_type == "tda":
            ax.set_ylabel(f"CV", fontsize=FONTSIZE_SMALL)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        # Apply consistent formatting
        format_plot_axes(
            ax, fig=fig, 
            model_type=model_type, 
            show_y_labels=(model_type == "tda"),
            align_ylabels=True
        )
        
        save_plot_dual_format(
            img_dir, filename,
            dpi=300, bbox_inches='tight',
            file_log_name=f"RMSD-{cv_dim}"
        )
        plt.close()

def plot_dssp_full_violin_analysis(
    cv,
    dssp_data,
    model_type,
    img_dir,
    date=None,
):
    """Plot violin plots for CV distribution by full DSSP secondary structure."""
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    
    # Get DSSP full data
    dssp_full = dssp_data['dssp_full']
    residue_indices = dssp_data['residue_indices']
    
    if dssp_full is None:
        print("DSSP full data not available, skipping violin analysis")
        return
    
    for cv_dim in range(MLCV_DIM):
        filename = f"dssp-full-violin-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
            
        print(f"> Plotting DSSP full violin analysis for {model_type}")
        
        simplified_mapping = get_dssp_simplified_mapping()
        ss_types = np.unique(dssp_full)
        ss_cv_data = {}
        
        for ss_type in ss_types:
            cv_values = []
            for frame_idx in range(len(dssp_full)):
                for res_idx in range(dssp_full.shape[1]):
                    if dssp_full[frame_idx, res_idx] == ss_type:
                        cv_values.append(cv[frame_idx, cv_dim])
            
            if len(cv_values) > 100:  # Minimum threshold for meaningful distribution
                ss_cv_data[ss_type] = np.array(cv_values)
        
        if len(ss_cv_data) < 2:
            print(f"Not enough secondary structure types for violin plot")
            continue
        
        # Define the order of simplified groups
        simplified_order = ['H', 'E', 'C']
        ordered_ss_types = []
        for simplified_type in simplified_order:
            for ss_type in sorted(ss_cv_data.keys()):
                if simplified_mapping.get(ss_type, 'C') == simplified_type:
                    ordered_ss_types.append(ss_type)
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=RECTANGLE_FIGSIZE)
        
        # Prepare data for violin plot in the new order
        violin_data = []
        violin_labels = []
        
        for ss_type in ordered_ss_types:
            if ss_type in ss_cv_data:
                violin_data.append(ss_cv_data[ss_type])
                simplified_type = simplified_mapping.get(ss_type, 'C')
                if ss_type == ' ':
                    ss_display = 'Coil'
                else:
                    ss_display = ss_type
                violin_labels.append(f'SS({simplified_type})-{ss_display}')
        
        # Create violin plot
        violin_parts = ax.violinplot(
            violin_data, positions=range(len(violin_data)), 
            showmeans=True, showmedians=False, showextrema=True,
        )
        
        # Customize violin plot
        format_violin_parts(violin_parts)
        
        ax.set_xticks(range(len(violin_labels)))
        ax.set_xticklabels(violin_labels, rotation=45)
        ax.set_ylabel(f'CV {cv_dim} Values')
        ax.set_title(f'CV {cv_dim} Distribution by DSSP Full Secondary Structure - {model_type.upper()}')
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(True, alpha=0.3)
        save_plot_dual_format(img_dir, filename, dpi=300, bbox_inches='tight',
            file_log_name=f"DSSP-Full-Violin-{cv_dim}"
        )
        plt.close()

def plot_dssp_simplified_violin_analysis(
    cv,
    dssp_data,
    model_type,
    img_dir,
    date=None,
):
    """Plot violin plots for CV distribution by simplified DSSP secondary structure."""
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    
    # Get DSSP simplified data
    dssp_simplified = dssp_data['dssp_simplified']
    residue_indices = dssp_data['residue_indices']
    
    if dssp_simplified is None:
        print("DSSP simplified data not available, skipping simplified violin analysis")
        return
    
    for cv_dim in range(MLCV_DIM):
        filename = f"dssp-simplified-violin-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
            
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
            
        print(f"> Plotting DSSP simplified violin analysis for {model_type}")
        
        # Collect CV values for each secondary structure type
        ss_types = np.unique(dssp_simplified)
        ss_cv_data = {}
        
        for ss_type in ss_types:
            cv_values = []
            for frame_idx in range(len(dssp_simplified)):
                for res_idx in range(dssp_simplified.shape[1]):
                    if dssp_simplified[frame_idx, res_idx] == ss_type:
                        cv_values.append(cv[frame_idx, cv_dim])
            
            if len(cv_values) > 100:  # Minimum threshold for meaningful distribution
                ss_cv_data[ss_type] = np.array(cv_values)
        
        if len(ss_cv_data) < 2:
            print(f"Not enough secondary structure types for violin plot")
            continue
        
        # Create violin plot
        fig = plt.figure(figsize=RECTANGLE_FIGSIZE)
        ax = fig.add_subplot(111)
        
        violin_data = []
        violin_labels = []
        for ss_type in sorted(ss_cv_data.keys()):
            violin_data.append(ss_cv_data[ss_type])
            violin_labels.append(ss_type)  # Simplified DSSP already has simple labels
        violin_parts = ax.violinplot(
            violin_data, positions=range(len(violin_data)), 
            showmeans=True, showmedians=False, showextrema=True
        )
        
        format_violin_parts(violin_parts)
        
        # ax.set_title(f'CV {cv_dim} Distribution by DSSP Simplified Secondary Structure - {model_type.upper()}')
        ax.set_xticks(range(len(violin_labels)))
        ax.set_xticklabels(violin_labels, rotation=45, fontsize=FONTSIZE_SMALL)
        if model_type == "tda":
            ax.set_ylabel(f'CV', fontsize=FONTSIZE_SMALL)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        format_plot_axes(
            ax, fig=fig, 
            model_type=model_type, 
            show_y_labels=(model_type == "tda"),
            align_ylabels=True
        )
        save_plot_dual_format(
            img_dir, filename,
            dpi=300, bbox_inches='tight',
            file_log_name=f"DSSP-Simplified-Violin-{cv_dim}"
        )
        plt.close()

def plot_dssp_cv_heatmap(
    cv,
    dssp_data,
    model_type,
    img_dir,
    date=None,
):
    """Plot heatmap for mean CV values by secondary structure and residue position."""
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    
    # Get DSSP data
    dssp_full = dssp_data['dssp_full']
    dssp_simplified = dssp_data['dssp_simplified']
    residue_indices = dssp_data['residue_indices']
    
    # Create list of available DSSP types
    dssp_types = []
    if dssp_full is not None:
        dssp_types.append(('full', dssp_full))
    if dssp_simplified is not None:
        dssp_types.append(('simplified', dssp_simplified))
    
    if not dssp_types:
        print("No DSSP data available, skipping CV heatmap analysis")
        return
    
    for cv_dim in range(MLCV_DIM):
        for dssp_type, dssp_array in dssp_types:
            if dssp_array is None:
                continue
                
            filename = f"dssp-{dssp_type}-cv-heatmap-cv{cv_dim}-{model_type}"
            if date:
                filename += f"_{date}"
                
            # Check for both PNG and PDF extensions
            if check_image_exists(img_dir, filename) or os.path.exists(os.path.join(img_dir, f"{filename}.pdf")):
                print(f"> Skipping {filename}.pdf - already exists")
                continue
                
            print(f"> Plotting DSSP {dssp_type} CV heatmap for {model_type}")
            
            # Get unique secondary structure types
            ss_types = np.unique(dssp_array)
            residue_range = range(dssp_array.shape[1])
            
            # Create heatmap data
            heatmap_data = np.full((len(ss_types), len(residue_range)), np.nan)
            ss_type_labels = []
            
            # Get simplified mapping for full DSSP
            if dssp_type == 'full':
                simplified_mapping = get_dssp_simplified_mapping()
            
            for ss_idx, ss_type in enumerate(ss_types):
                if dssp_type == 'full':
                    simplified_type = simplified_mapping.get(ss_type, 'C')
                    if ss_type == ' ':
                        ss_display = 'Coil'
                    else:
                        ss_display = ss_type
                    ss_name = f'SS({simplified_type})-{ss_display}'
                else:
                    ss_name = ss_type  # Simplified DSSP already has simple labels
                ss_type_labels.append(ss_name)
                
                for res_idx in residue_range:
                    # Collect CV values for this residue when it has this secondary structure
                    cv_values_for_residue = []
                    
                    for frame_idx in range(len(dssp_array)):
                        if dssp_array[frame_idx, res_idx] == ss_type:
                            cv_values_for_residue.append(cv[frame_idx, cv_dim])
                    
                    if len(cv_values_for_residue) > 10:  # Minimum threshold for meaningful average
                        heatmap_data[ss_idx, res_idx] = np.mean(cv_values_for_residue)
            
            # Create the heatmap with reduced spacing
            fig, ax = plt.subplots(1, 1, figsize=RECTANGLE_FIGSIZE)
            
            # Plot mean CV values heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
            
            # Add text annotations with mean CV values
            for ss_idx in range(len(ss_type_labels)):
                for res_idx in range(len(residue_range)):
                    value = heatmap_data[ss_idx, res_idx]
                    if not np.isnan(value):
                        # Choose text color based on background color intensity
                        text_color = 'white' if abs(value) > np.nanmax(np.abs(heatmap_data)) * 0.5 else 'black'
                        ax.text(res_idx, ss_idx, f'{value:.2f}', 
                               ha='center', va='center', color=text_color, fontsize=8, weight='bold')
            
            ax.set_xlabel('Residue Index', fontsize=12)
            ax.set_ylabel('Secondary Structure', fontsize=12)
            ax.set_title(f'Mean CV {cv_dim} Values by DSSP {dssp_type.title()} Secondary Structure and Residue - {model_type.upper()}', fontsize=13)
            ax.set_yticks(range(len(ss_type_labels)))
            ax.set_yticklabels(ss_type_labels, fontsize=10)
            
            # Set x-axis labels to show actual residue numbers
            if residue_indices is not None:
                n_ticks = min(len(residue_indices), 10)  # Show max 10 ticks
                tick_positions = np.linspace(0, len(residue_indices)-1, n_ticks, dtype=int)
                tick_labels = [str(residue_indices[i]) for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=10)
            
            # Add colorbar with reduced padding
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f'Mean CV {cv_dim} Value', fontsize=11)
            
            # Save in both formats with tight layout and reduced spacing
            save_plot_dual_format(
                img_dir, filename,
                bbox_inches='tight', pad_inches=0.1,
                file_log_name=f"DSSP-CV-Heatmap-{cv_dim}"
            )
            plt.close()

def plot_per_residue_violin_analysis(
    cv,
    dssp_data,
    model_type,
    molecule,
    img_dir,
    date=None,
):
    """Plot per-residue secondary structure violin plots for selected residues."""
    os.makedirs(img_dir, exist_ok=True)
    MLCV_DIM = cv.shape[1]
    
    # Get DSSP simplified data
    dssp_simplified = dssp_data['dssp_simplified']
    residue_indices = dssp_data['residue_indices']
    
    if dssp_simplified is None:
        print("DSSP simplified data not available, skipping per-residue violin analysis")
        return
    
    # Define selected residues for different molecules
    if molecule == "CLN025":
        selected_residues = [1, 2, 7, 8]
    elif molecule == "2JOF":
        selected_residues = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    elif molecule == "1FME":
        selected_residues = [4, 5, 6, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    elif molecule == "GTT":
        selected_residues = [7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 27, 28]
    # elif molecule == "2F4K":
    #     selected_residues = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    # elif molecule == "NTL9":
    #     selected_residues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    else:
        print(f"No specific residues defined for molecule {molecule}, skipping per-residue analysis")
        return
    
    # Limit to maximum plots and available residues
    max_plots = 24
    available_residues = min(dssp_simplified.shape[1], len(selected_residues))
    selected_residues = selected_residues[:min(max_plots, available_residues)]
    
    for cv_dim in range(MLCV_DIM):
        filename = f"dssp-per-residue-violin-cv{cv_dim}-{model_type}"
        if date:
            filename += f"_{date}"
        if check_image_exists(img_dir, filename):
            print(f"> Skipping {filename}.png - already exists")
            continue
            
        print(f"> Plotting per-residue DSSP violin analysis for {model_type} {molecule}")
        
        # Collect CV values for each residue and secondary structure type
        residue_data = {}
        for residue_idx in selected_residues:
            if residue_idx >= dssp_simplified.shape[1]:
                continue
            ss_data = {}
            ss_types = np.unique(dssp_simplified[:, residue_idx])
            for ss_type in ss_types:
                cv_values = []
                for frame_idx in range(len(dssp_simplified)):
                    if dssp_simplified[frame_idx, residue_idx] == ss_type:
                        cv_values.append(cv[frame_idx, cv_dim])
                
                if len(cv_values) > 50:  # Minimum threshold for meaningful distribution
                    ss_data[ss_type] = np.array(cv_values)
            if len(ss_data) >= 2:  # Need at least 2 secondary structure types
                residue_data[residue_idx] = ss_data
        if len(residue_data) == 0:
            print(f"No residues found with sufficient secondary structure diversity for analysis")
            continue
        
        # Calculate grid size
        n_cols = 4
        n_rows = (len(residue_data) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        for residue_idx in sorted(residue_data.keys()):
            ax = axes[plot_idx]
            ss_data = residue_data[residue_idx]
            ss_types_present = sorted(ss_data.keys())
            cv_data_list = [ss_data[ss_type] for ss_type in ss_types_present]
            
            # Create violin plot
            violin_parts = ax.violinplot(
                cv_data_list, positions=range(len(ss_types_present)),
                showmeans=True, showmedians=False, showextrema=True,
            )
            
            format_violin_parts(violin_parts)
            
            # ax.set_title(f'Residue {residue_label}')
            ss_labels = ['Coil' if ss == ' ' else ss for ss in ss_types_present]
            ax.set_xlabel(f'Secondary Structure {residue_idx}', fontsize=FONTSIZE_SMALL)
            if model_type == "tda":
                ax.set_ylabel(f'CV {cv_dim} Value', fontsize=FONTSIZE_SMALL)
                ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            if residue_indices is not None and residue_idx < len(residue_indices):
                residue_label = residue_indices[residue_idx]
            else:
                residue_label = residue_idx + 1
            ax.set_xticks(range(len(ss_types_present)))
            ax.set_xticklabels(ss_labels, fontsize=FONTSIZE_SMALL)
            format_plot_axes(
                ax, fig=fig, 
                model_type=model_type, 
                show_y_labels=(model_type == "tda"),
                align_ylabels=True
            )
            plot_idx += 1
        
        for idx in range(len(residue_data), len(axes)):
            axes[idx].set_visible(False)
        
        # plt.suptitle(f'CV {cv_dim} Distribution by Secondary Structure for Each Residue - {model_type.upper()}', fontsize=14)
        save_plot_dual_format(
            img_dir, filename,
            dpi=300, bbox_inches='tight',
            file_log_name=f"DSSP-Per-Residue-Violin-{cv_dim}"
        )
        plt.close()

def plot_dssp_composition_heatmap(
    dssp_data,
    img_dir
):
    """Plot heatmap for secondary structure composition by residue position."""
    os.makedirs(img_dir, exist_ok=True)
    
    # Get DSSP data
    dssp_full = dssp_data['dssp_full']
    dssp_simplified = dssp_data['dssp_simplified']
    residue_indices = dssp_data['residue_indices']
    
    # Create list of available DSSP types
    dssp_types = []
    if dssp_full is not None:
        dssp_types.append(('full', dssp_full))
    if dssp_simplified is not None:
        dssp_types.append(('simplified', dssp_simplified))
    
    if not dssp_types:
        print("No DSSP data available, skipping composition heatmap analysis")
        return
    
    for dssp_type, dssp_array in dssp_types:
        if dssp_array is None:
            continue
            
        filename = f"dssp-{dssp_type}-composition-heatmap"
        if check_image_exists(img_dir, filename) or os.path.exists(os.path.join(img_dir, f"{filename}.pdf")):
            print(f"> Skipping {filename} - already exists (composition is method-independent)")
            continue
            
        print(f"> Plotting DSSP {dssp_type} composition heatmap (method-independent)")
        
        # Get unique secondary structure types
        ss_types = np.unique(dssp_array)
        ss_type_labels = []
        
        # Get simplified mapping for full DSSP
        if dssp_type == 'full':
            simplified_mapping = get_dssp_simplified_mapping()
        
        for ss_type in ss_types:
            if dssp_type == 'full':
                simplified_type = simplified_mapping.get(ss_type, 'C')
                if ss_type == ' ':
                    ss_display = 'Coil'
                else:
                    ss_display = ss_type
                ss_name = f'SS({simplified_type})-{ss_display}'
            else:
                ss_name = ss_type  # Simplified DSSP already has simple labels
            ss_type_labels.append(ss_name)
        
        # Create secondary structure composition data
        ss_composition = np.zeros((len(ss_types), dssp_array.shape[1]))
        
        for res_idx in range(dssp_array.shape[1]):
            for frame_idx in range(len(dssp_array)):
                ss_type = dssp_array[frame_idx, res_idx]
                ss_idx = list(ss_types).index(ss_type)
                ss_composition[ss_idx, res_idx] += 1
        
        # Normalize to get proportions
        ss_composition = ss_composition / len(dssp_array)
        
        # Create the heatmap with reduced spacing
        fig, ax = plt.subplots(1, 1, figsize=BIG_RECTANGLE_FIGSIZE)
        
        # Plot composition
        im = ax.imshow(ss_composition, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Add text annotations with composition values
        for ss_idx in range(len(ss_type_labels)):
            for res_idx in range(ss_composition.shape[1]):
                value = ss_composition[ss_idx, res_idx]
                # Choose text color based on background color intensity
                text_color = 'white'
                ax.text(
                    res_idx, ss_idx, f'{value:.2f}', 
                    ha='center', va='center', color=text_color, fontsize=8, weight='bold'
                )
        
        ax.set_xlabel('Residue Index', fontsize=12)
        ax.set_ylabel('Secondary Structure', fontsize=12)
        ax.set_title(f'DSSP {dssp_type.title()} Secondary Structure Composition by Residue', fontsize=13)
        ax.set_yticks(range(len(ss_type_labels)))
        ax.set_yticklabels(ss_type_labels, fontsize=10)
        
        # Set x-axis labels to show actual residue numbers
        if residue_indices is not None:
            n_ticks = min(len(residue_indices), 10)  # Show max 10 ticks
            tick_positions = np.linspace(0, len(residue_indices)-1, n_ticks, dtype=int)
            tick_labels = [str(residue_indices[i]) for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=10)
        
        # Add colorbar with reduced padding
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Proportion of Frames', fontsize=11)
        save_plot_dual_format(
            img_dir, filename,
            bbox_inches='tight', pad_inches=0.1,
            file_log_name=f"DSSP-Composition-Heatmap-{dssp_type}"
        )
        plt.close()

def analyze_correlations(
    cv,
    committor_value,
    tica_data,
    bond_num,
    molecule,
    model_type,
):
    print(f"Analyzing correlations for {model_type} {molecule}")
    MLCV_DIM = cv.shape[1]
    
    print(f"\n{model_type.upper()} - {molecule} Correlation Analysis")
    print("=" * 60)

    correlation_results = []
    for cv_dim in range(MLCV_DIM):
        cv_values = cv[:, cv_dim]
        correlation_p, p_value_p = pearsonr(committor_value, cv_values)
        correlation_s, p_value_s = spearmanr(committor_value, cv_values)
        
        correlation_results.append({
            'CV_Dimension': cv_dim,
            'Pearson_Correlation': correlation_p,
            'Pearson_P_Value': p_value_p,
            'Spearman_Correlation': correlation_s,
            'Spearman_P_Value': p_value_s,
            'Pearson_Strength': 'Strong' if abs(correlation_p) > 0.7 else 'Moderate' if abs(correlation_p) > 0.3 else 'Weak',
            'Spearman_Strength': 'Strong' if abs(correlation_s) > 0.7 else 'Moderate' if abs(correlation_s) > 0.3 else 'Weak'
        })
        
        print(f"CV {cv_dim} vs Committor:")
        print(f"  Pearson correlation: {correlation_p:.6f} (p = {p_value_p:.2e}) - {correlation_results[-1]['Pearson_Strength']}")
        print(f"  Spearman correlation: {correlation_s:.6f} (p = {p_value_s:.2e}) - {correlation_results[-1]['Spearman_Strength']}")

    # Additional correlations
    tica_x = tica_data[:, 0]
    tica_y = tica_data[:, 1]
    
    corr_tica_x_p, p_tica_x_p = pearsonr(committor_value, tica_x)
    corr_tica_y_p, p_tica_y_p = pearsonr(committor_value, tica_y)
    corr_tica_x_s, p_tica_x_s = spearmanr(committor_value, tica_x)
    corr_tica_y_s, p_tica_y_s = spearmanr(committor_value, tica_y)
    
    print(f"\nCommittor vs TICA-1:")
    print(f"  Pearson: r = {corr_tica_x_p:.6f}, p = {p_tica_x_p:.2e}")
    print(f"  Spearman: ρ = {corr_tica_x_s:.6f}, p = {p_tica_x_s:.2e}")
    print(f"Committor vs TICA-2:")
    print(f"  Pearson: r = {corr_tica_y_p:.6f}, p = {p_tica_y_p:.2e}")
    print(f"  Spearman: ρ = {corr_tica_y_s:.6f}, p = {p_tica_y_s:.2e}")
    wandb.log({
        f"tica/pearson_r": corr_tica_x_p,
        f"tica/pearson_p": p_tica_x_p,
        f"tica/spearman_rho": corr_tica_x_s,
        f"tica/spearman_p": p_tica_x_s,
        f"tica/pearson_r_y": corr_tica_y_p,
        f"tica/pearson_p_y": p_tica_y_p,
        f"tica/spearman_rho_y": corr_tica_y_s,
        f"tica/spearman_p_y": p_tica_y_s,
    })
    
    if bond_num is not None:
        corr_bond_p, p_bond_p = pearsonr(committor_value, bond_num)
        corr_bond_s, p_bond_s = spearmanr(committor_value, bond_num)
        print(f"Committor vs Bond Number:")
        print(f"  Pearson: r = {corr_bond_p:.6f}, p = {p_bond_p:.2e}")
        print(f"  Spearman: ρ = {corr_bond_s:.6f}, p = {p_bond_s:.2e}")
        wandb.log({
            f"bond/pearson_r": corr_bond_p,
            f"bond/pearson_p": p_bond_p,
            f"bond/spearman_rho": corr_bond_s,
            f"bond/spearman_p": p_bond_s,
        })


# MAIN
def main():
    parser = argparse.ArgumentParser(description='Run CV analysis for different model types')
    parser.add_argument('--model_type', choices=['ours', 'tda', 'tica', 'tae', 'vde', 'all'], default='all', help='Model type to analyze')
    parser.add_argument('--molecule', choices=['CLN025', '2JOF', '2F4K', '1FME', 'GTT', 'NTL9', 'partial','all'], default='CLN025', help='Molecule to analyze')
    parser.add_argument('--date', type=str, default=None, help='Date string for MLCV model (only used for mlcv)')
    parser.add_argument('--img_dir', type=str, default='/home/shpark/prj-mlcv/lib/bioemu/img/debug', help='Directory to save images')
    parser.add_argument('--cuda_device', type=int, default=None, help='CUDA device ID to use (e.g., 0, 1). If not specified, auto-detect available device')
    parser.add_argument('--plot_3d', type=bool, default=False, help='Plot 3D scatter plot')
    # parser.add_argument('--dssp_analysis', type=bool, default=False, help='Perform DSSP analysis')
    args = parser.parse_args()
    
    # Override global CUDA_DEVICE if specified
    global CUDA_DEVICE
    if args.cuda_device is not None:
        CUDA_DEVICE = args.cuda_device
        print(f"Using specified CUDA device: cuda:{CUDA_DEVICE}")
    
    # Create image directory if it doesn't exist
    os.makedirs(args.img_dir, exist_ok=True)
    model_types = ['ours', 'tda', 'tica', 'tae', 'vde'] if args.model_type == 'all' else [args.model_type]
    if args.molecule == 'partial':
        molecules = ['1FME', '2F4K', 'GTT', 'NTL9']
    elif args.molecule == 'all':
        molecules = ['CLN025', '2JOF', '2F4K', '1FME', 'GTT', 'NTL9']
    else:
        molecules = [args.molecule]
    
    for molecule in molecules:
        img_dir_mol = os.path.join(args.img_dir, molecule)
        os.makedirs(img_dir_mol, exist_ok=True)
        for model_type in model_types:
            try:
                print(f"\n{'='*60}")
                print(f"Running analysis for {model_type.upper()} - {molecule}")
                print(f"{'='*60}")
                wandb.init(
                    project="cv-analysis",
                    name=f"{model_type}-{molecule}",
                    config=vars(args),
                )
                
                # Load model and data
                mlcv_model, tica_wrapper, committor_model, pos_path, cad_path, cad_switch_path = load_model_and_data(
                    model_type, molecule, args.date
                )
                print(f"Loaded model: {model_type}")
                
                # Load reference structure
                reference_pdb_path = f"./opes/data/{molecule}/folded.pdb"
                reference_cad = None
                if molecule in ["CLN025","2JOF","2F4K","1FME","GTT","NTL9"] and os.path.exists(reference_pdb_path):
                    reference_cad = load_reference_structure(reference_pdb_path, tica_wrapper)
                    print(f"Loaded reference structure from {reference_pdb_path}")
                else:
                    print(f"Reference structure not given, CV sign not aligned")
                
                # Compute CV values using batch processing
                cv = compute_cv_values(
                    mlcv_model,
                    cad_path,
                    model_type,
                    molecule=molecule,
                    reference_cad=reference_cad,
                    batch_size=10000,
                    device=CUDA_DEVICE
                )
                print(f"CV shape: {cv.shape}")
                print(f"CV range: {cv.max():.4f} to {cv.min():.4f}")
                # plot_cv_histogram(cv, model_type, molecule, img_dir_mol, args.date)
                
                # Compute TICA coordinates
                tica_coord_path = f"./opes/data/{molecule.upper()}/tica_lag10_coord.npy"
                if os.path.exists(tica_coord_path):
                    print(f"> Using cached TICA coordinates from {tica_coord_path}")
                    tica_data = np.load(tica_coord_path)
                else:
                    if molecule == "CLN025":
                        cad_switch_torch = torch.load(cad_switch_path).to(CUDA_DEVICE)
                        cad_switch_torch = cad_switch_torch.cpu() if cad_switch_torch.device.type == "cuda" else cad_switch_torch
                        tica_data = tica_wrapper.transform(cad_switch_torch.numpy())
                    else:
                        cad_torch = torch.load(cad_path).to(CUDA_DEVICE)
                        cad_torch = cad_torch.cpu() if cad_torch.device.type == "cuda" else cad_torch
                        tica_data = tica_wrapper.transform(cad_torch.numpy())
                print(f"TICA shape: {tica_data.shape}")
                plot_tica_cv_analysis(cv, tica_data, model_type, molecule, img_dir_mol, args.date, args.plot_3d)
                if cv.shape[1] > 1:
                    plot_cv_tica_analysis(cv, tica_data, model_type, molecule, img_dir_mol, args.date)
                
                # RMSD analysis
                plot_rmsd_analysis(cv, molecule, model_type, img_dir_mol, args.date)
                
                # DSSP analysis - load and filter data once
                print(f"\nLoading DSSP data for {molecule}...")
                dssp_data = load_and_filter_dssp_data(molecule)
                plot_per_residue_violin_analysis(cv, dssp_data, model_type, molecule, img_dir_mol, args.date)
                plot_dssp_simplified_violin_analysis(cv, dssp_data, model_type, img_dir_mol, args.date)
                # plot_dssp_full_violin_analysis(cv, dssp_data, model_type, img_dir_mol, args.date)
                # plot_dssp_cv_heatmap(cv, dssp_data, model_type, img_dir_mol, args.date)
                # plot_dssp_composition_heatmap(dssp_data, img_dir_mol)
                
                # Correlation analysis
                if molecule == "CLN025":
                    pos_torch = torch.load(pos_path).to(CUDA_DEVICE)
                    cad_torch = torch.load(cad_path).to(CUDA_DEVICE)
                    cad_switch_torch = torch.load(cad_switch_path).to(CUDA_DEVICE)
                    plot_committor_analysis(cv, cad_torch, committor_model, model_type, molecule, img_dir_mol, args.date)
                    plot_bond_analysis(cv, pos_torch, tica_wrapper, molecule, model_type, img_dir_mol, args.date)
                    
                    bond_path = "/home/shpark/prj-mlcv/lib/bioemu/opes/data/CLN025/bond_num.npy"
                    if os.path.exists(bond_path):
                        print(f"> Loaded bond number from {bond_path}")
                        bond_num = np.load(bond_path)
                    else:
                        print(f"> No bond number found at {bond_path}, computing...")
                        dummy_pdb = tica_wrapper.pdb
                        dummy_pdb.xyz = pos_torch.cpu().detach().numpy()
                        _, bond_num = foldedness_by_hbond_distance(dummy_pdb)
                        np.save(bond_path, bond_num)
                    committor_value = committor_model(cad_torch.to(CUDA_DEVICE))
                    committor_value = committor_value.cpu().detach().numpy().flatten()
                    analyze_correlations(cv, committor_value, tica_data, bond_num, molecule, model_type)
                
                print(f"\nCompleted analysis for {model_type.upper()} - {molecule}. Plots saved to {img_dir_mol}")
                wandb.finish()
                
            except Exception as e:
                print(f"Error during analysis for {model_type.upper()} - {molecule}: {e}")
                wandb.finish()
                continue

if __name__ == "__main__":
    main()
