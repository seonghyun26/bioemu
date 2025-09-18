import os
import sys
import torch
import torch.utils.data
import hydra
import argparse
import subprocess
import pickle
import wandb
import logging
import numexpr
import numpy as np
import mdtraj as md

from tqdm import tqdm
from pprint import pformat
from itertools import combinations
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections
import matplotlib.lines
import matplotlib.patches
import matplotlib.image
from matplotlib.colors import LogNorm

from adaptive_sampling.processing_tools import mbar
from adaptive_sampling.processing_tools.utils import DeltaF_fromweights

from src import *
from src.constant import *

os.environ["NUMEXPR_MAX_THREADS"] = "32"   # or any value you like
np.NaN = np.nan
blue = (70 / 255, 110 / 255, 250 / 255)
R = 0.008314462618  # kJ/mol/K
logger = logging.getLogger(__name__)
FONTSIZE = 20
FONTSIZE_SMALL = 16
LINEWIDTH = 1.5
mpl.rcParams['axes.linewidth'] = LINEWIDTH  # default is 0.8
mpl.rcParams['xtick.major.width'] = LINEWIDTH
mpl.rcParams['ytick.major.width'] = LINEWIDTH
mpl.rcParams['xtick.minor.width'] = LINEWIDTH
mpl.rcParams['ytick.minor.width'] = LINEWIDTH


# Load components
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
    ax.spines['left'].set_linewidth(LINEWIDTH)
    ax.spines['bottom'].set_linewidth(LINEWIDTH)
    
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
            ax.set_ylabel("")
            # Hide y-axis labels for non-TDA models unless explicitly requested
            if show_y_labels:
                ax.tick_params(axis='both', labelsize=fontsize)
            else:
                ax.tick_params(axis='both', labelsize=fontsize, labelleft=False)
    
    # Align y-labels if requested and figure is provided
    if align_ylabels and fig is not None:
        fig.align_ylabels(ax)


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


def save_plot_dual_format(
    img_dir,
    filename,
    dpi=200,
    bbox_inches='tight',
    pad_inches=0.1,
    rasterized=True,
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
    
    Returns:
        bool: True if files were saved, False if they already existed
    """
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, "pdf"), exist_ok=True)
    
    png_path = os.path.join(img_dir, f"{filename}.png")
    pdf_path = os.path.join(img_dir, f"pdf/{filename}.pdf")
    
    # Check if both files already exist
    # if check_image_exists(img_dir, filename):
    #     print(f"> Skipping {filename} - both PNG and PDF already exist")
    #     return False
    
    # Save in both formats
    try:
        # Save as PNG
        # if not os.path.exists(png_path):
        plt.savefig(
            png_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        print(f">> Saved {png_path}")
        
        # Save as PDF
        # if not os.path.exists(pdf_path):
        if rasterized:
            rasterize_plot_elements()
        plt.savefig(
            pdf_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        wandb.save(pdf_path)
        print(f">> Saved {pdf_path}")

        return True
        
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False


def gmx_process_trajectory(
    cfg,
    data_dir: Path,
    log_dir: Path,
    analysis_dir: Path,
    max_seed: int
):
    """Post process trajectory."""
    
    pbar = tqdm(
        range(max_seed + 1),
        desc="Post processing trajectory"
    )
    data_dir = data_dir
    for seed in pbar:
        trj_save_path = f"{analysis_dir}/{seed}_tc.xtc"
        # if os.path.exists(trj_save_path):
        #     print(f"✓ trjconv file already exists: {trj_save_path}")
        #     continue
        
        # else:
        cmd = [
            "gmx", "trjconv",
            "-f", f"{log_dir}/{seed}.xtc",
            "-s", f"{data_dir}/protein.tpr",
            "-pbc", "mol",
            "-o", trj_save_path,
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            process = subprocess.run(
                cmd,
                input="0\n0\n",
                capture_output=True,
                text=True
            )
            if process.returncode == 0:
                print(f"✓ gmx trjconv completed successfully, seed {seed}")
                print(f"Created trjconv file: {trj_save_path}")
            else:
                print(f"✗ gmx trjconv failed with return code {process.returncode}, seed {seed}")
            if process.stdout:
                print(f"STDOUT:, seed {seed}:", process.stdout)
            if process.stderr:
                print(f"GROMACSOUT:, seed {seed}:", process.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"gmx trjconv failed, seed {seed}: {e}")
    
    
def gmx_process_energy(
    log_dir: Path,
    analysis_dir: Path,
    max_seed: int,
):
    """Compute energy from trajectory data."""
    
    # GROMACS command for energy calculation
    pbar = tqdm(
        range(max_seed + 1),
        desc="Computing energy"
    )
    for seed in pbar:
        edr_save_path = f"{analysis_dir}/{seed}.xvg"
        # if os.path.exists(edr_save_path):
        #     print(f"✓ energy file already exists: {edr_save_path}")
        #     continue
        
        # else:
        cmd = [
            "gmx", "energy",
            "-f", f"{log_dir}/{seed}.edr", 
            "-o", f"{analysis_dir}/{seed}.xvg",
            '-xvg', 'none'
        ]

        print(f"Running command: {' '.join(cmd)}")
        try:
            process = subprocess.run(
                cmd,
                input="16 17 9 0\\n",
                capture_output=True,
                text=True
            )
            if process.returncode == 0:
                print(f"✓ gmx energy completed successfully, seed {seed}")
                print(f"Created energy file: {analysis_dir}/{seed}.xvg")
            else:
                print(f"✗ gmx energy failed with return code {process.returncode}, seed {seed}")
            if process.stdout:
                print(f"STDOUT:, seed {seed}:", process.stdout)
            if process.stderr:
                print(f"GROMACSOUT:, seed {seed}:", process.stderr)
            
        except subprocess.CalledProcessError as e:
            print(f"gmx energy failed, seed {seed} : {e}")



def compute_cv_values(
    cfg,
    max_seed: int,
    batch_size=10000,
):
    """
    Compute reference CV values using batch processing to prevent GPU memory issues.
    
    Args:
        cfg: Configuration object containing model and data paths
        batch_size: Size of batches for processing (default: 10000)
    
    Returns:
        tuple: cv_values
    """
    cv_path = f"./data/{cfg.molecule.upper()}/{cfg.method}_mlcv.npy"
    if os.path.exists(cv_path):
        print(f"> Using cached CV values from {cv_path}")
        cv = np.load(cv_path)
        return cv
    
    else:
        print(f"> Computing CV values")
        try:
            base_simulation_dir = Path(f"{os.getcwd()}/simulations") / cfg.molecule / cfg.method
            seed_dir = base_simulation_dir / f"{cfg.date}/0"
            jit_files = list(seed_dir.glob("*-jit.pt"))
            if not jit_files:
                raise FileNotFoundError(f"No -jit.pt files found in {seed_dir}")
            mlcv_model_path = jit_files[0]
            mlcv_model = torch.jit.load(mlcv_model_path)
            mlcv_model.eval()
            mlcv_model.to("cuda:0")
            cad_full_path = f"./data/{cfg.molecule.upper()}/cad.pt"
            cad_full = torch.load(cad_full_path)
            cad_full = cad_full.to("cuda:0")
            print(f"> Using model file: {mlcv_model_path}")
        except Exception as e:
            logger.warning(f"Error loading model: {e}")
            return None
        
        # Compute CVs in batches to prevent GPU memory issues
        try :
            dataset = TensorDataset(cad_full)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                sample_batch = next(iter(dataloader))[0]
                sample_output = mlcv_model(sample_batch)
                output_dim = sample_output.shape[1]
            
            cv_batches = torch.zeros((len(cad_full), output_dim)).to("cuda:0")
            with torch.no_grad():
                for batch_idx, (batch_data,) in enumerate(tqdm(
                    dataloader,
                    desc="Computing CV values",
                    total=len(dataloader),
                    leave=False,
                )):
                    batch_cv = mlcv_model(batch_data)
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_cv.shape[0]
                    cv_batches[start_idx:end_idx] = batch_cv
            
            cv = cv_batches.detach().cpu().numpy()
            print(f"> CV computation complete. Shape: {cv.shape}")
        except Exception as e:
            logger.warning(f"Error computing CV values: {e}")
            return None
        
        return cv


def plot_pmf(
    cfg,
    sigma: float,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
    reference_cvs: np.ndarray,
):
    print(f"> Computing PMF")
    equil_temp = cfg.analysis.equil_temp
    filename = "pmf"
    # if check_image_exists(str(analysis_dir), filename):
    #     print(f"✓ PMF plot already exists: {filename}")
    #     wandb.log({
    #         "pmf": wandb.Image(str(analysis_dir / f"{filename}.png")),
    #     })
    #     return
    
    # else:
    
    all_cv_grids = []
    all_pmfs = []
    
    # Check CV range
    print(f"> Computing OPES PMF CVs range")
    cv_mins = []
    cv_maxs = []
    pbar = tqdm(
        range(max_seed + 1),
        desc="Checking CV range"
    )
    for seed in pbar:
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        if not colvar_file.exists():
            logger.warning(f"COLVAR file not found: {colvar_file}")
            continue
        colvar_data = np.genfromtxt(colvar_file, skip_header=1)
        cv = colvar_data[:, 1]
        cv_grid = np.arange(cv.min(), cv.max() + sigma / 2, sigma)
        all_cv_grids.append(cv_grid)
        cv_mins.append(cv.min())
        cv_maxs.append(cv.max())
    cv_grid_min = np.min(cv_mins) if len(cv_mins) > 0 else 0.0
    cv_grid_max = np.max(cv_maxs) if len(cv_maxs) > 0 else 0.0
    cv_grid_min = min(cv_grid_min, -1.0)
    cv_grid_max = max(cv_grid_max, 1.0)
    cv_grid = np.arange(cv_grid_min - sigma / 2, cv_grid_max + sigma / 2, sigma)
    
    # Compute PMF
    pbar = tqdm(
        range(max_seed + 1),
        desc="Computing OPES PMF"
    )
    for seed in pbar:
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        colvar_data = np.genfromtxt(colvar_file, skip_header=1)
        cv = colvar_data[:, 1]
        bias = colvar_data[:, 2]
        beta = 1.0 / (R * equil_temp)
        W = np.exp(beta * bias)  
        pmf, _ = mbar.pmf_from_weights(
            cv_grid,
            cv,
            W,
            equil_temp=equil_temp
        )
        pmf -= pmf.min()
        all_pmfs.append(pmf)
    all_pmfs = np.array(all_pmfs)
    mean_pmf = np.mean(all_pmfs, axis=0)
    std_pmf = np.std(all_pmfs, axis=0)
    
    # Compute reference PMF
    print(f"> Computing reference PMF")
    reference_pmf, _ = mbar.pmf_from_weights(
        cv_grid,
        reference_cvs,
        weights=np.ones_like(reference_cvs),
        equil_temp=equil_temp
    )
    reference_pmf -= reference_pmf.min()
    reference_mask = ~np.isnan(reference_pmf)
    mean_pmf_mask = ~np.isnan(mean_pmf)
    pmf_mask = reference_mask & mean_pmf_mask
    pmf_mae = np.mean(np.abs(mean_pmf[pmf_mask] - reference_pmf[pmf_mask]))

    print(f"> Plotting PMF")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(
        cv_grid, mean_pmf,
        color=blue, linewidth=4,
        zorder=4,
    )
    ax.fill_between(
        cv_grid, mean_pmf - std_pmf, mean_pmf + std_pmf,
        alpha=0.2, color=blue, linewidth=1,
    )
    for idx, pmf in enumerate(all_pmfs):
        ax.plot(
            cv_grid, pmf,
            color=blue, linewidth=2, alpha=0.2,
            label=f"OPES {idx}",
            zorder=2,
        )
    ax.plot(
        cv_grid, reference_pmf,
        color=COLORS[1], linewidth=3, linestyle="--",
        label="Reference",
        zorder=6,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=FONTSIZE_SMALL)
    ax.set_xlabel("CV", fontsize=FONTSIZE_SMALL)
    ax.set_ylabel("PMF [kJ/mol]", fontsize=FONTSIZE_SMALL)
    plt.grid(True, alpha=0.3)
    format_plot_axes(
        ax, fig=fig, 
        model_type=cfg.method, 
        show_y_labels=True,
        align_ylabels=True
    )
    save_plot_dual_format(str(analysis_dir), filename, dpi=300, bbox_inches="tight")
    logger.info(f"PMF plot saved to {analysis_dir}")
    wandb.log({
        "pmf": wandb.Image(str(analysis_dir / f"{filename}.png")),
        "pmf_mae": round(pmf_mae, 2),
        "pmf_std": round(std_pmf[~np.isnan(std_pmf)].mean(), 2)
    })
    plt.close()
    
    return


def plot_free_energy_curve(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
    reference_cvs: np.ndarray,
):
    filename = "free_energy_curve"
    # if check_image_exists(str(analysis_dir), filename):
    #     print(f"✓ Free energy curve plot already exists: {filename}")
    #     wandb.log({
    #         "free_energy_curve": wandb.Image(str(analysis_dir / f"{filename}.png")),
    #     })
    #     return
    
    # else:
    print(f"> Computing free energy curve")
    # ns_per_step = 0.004
    # skip_steps = 50000
    skip_steps = cfg.analysis.skip_steps
    unit_steps = cfg.analysis.unit_steps
    equil_temp = cfg.analysis.equil_temp
    all_times = []
    all_cvs = []
    all_delta_fs = []
    
    # Compute Delta F
    print(f"> Computing Delta F")
    pbar = tqdm(
        range(max_seed + 1),
        desc="Computing Delta F"
    )
    for seed in pbar:
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        try:
            colvar_data = np.genfromtxt(colvar_file, skip_header=1)
            time = colvar_data[:, 0]
            cv = colvar_data[:, 1]
            bias = colvar_data[:, 2]
            total_steps = len(colvar_data)
            step_grid = np.arange(
                skip_steps + unit_steps, total_steps, unit_steps
            )
            cv_thresh = [
                -2, 0, 2
            ]
            beta = 1.0 / (R * equil_temp)
            W = np.exp(beta * bias)  
            all_cvs.append(cv)
            time_steps = time[step_grid] * 0.001
            if time_steps.shape[0] == 0:
                step_grid_without_skip_steps = np.arange(
                    unit_steps, total_steps, unit_steps
                )
                time_steps = time[step_grid_without_skip_steps] * 0.001
            all_times.append(time_steps)
            
            Delta_Fs = []
            for current_step in tqdm(step_grid):
                cv_t = cv[skip_steps:current_step]
                W_t = W[skip_steps:current_step]
                    
                Delta_F = DeltaF_fromweights(
                    xi_traj=cv_t,
                    weights=W_t,
                    cv_thresh=cv_thresh,
                    T=equil_temp,
                )
                Delta_Fs.append(Delta_F)
            Delta_Fs = np.array(Delta_Fs)
            all_delta_fs.append(Delta_Fs)
            
        except Exception as e:
            logger.warning(f"Error processing data for seed {seed}: {e}")
            continue
    
    # Compute mean and std of delta F
    print(f"> Computing mean and std of delta F")
    max_len = max([len(x) for x in all_delta_fs])
    all_delta_fs_padded = np.full((len(all_delta_fs), max_len), np.nan, dtype=float)
    for i, x in enumerate(all_delta_fs):
        all_delta_fs_padded[i, :len(x)] = x
    idx_longest = int(np.argmax([len(t) for t in all_times]))
    time_axis = all_times[idx_longest]
    print(f"Time axis shape: {time_axis.shape}")
    print(f"Time axis last: {time_axis[-1]}")
    
    # Compute mean/std ignoring NaNs, but only keep columns where at least one value is present
    has_data = (~np.isnan(all_delta_fs_padded)) & (~np.isinf(all_delta_fs_padded))
    valid_values = np.where(has_data, all_delta_fs_padded, np.nan)
    mean_delta_fs = np.nanmean(valid_values, axis=0)
    std_delta_fs  = np.nanstd(valid_values,  axis=0)
    print(f"Mean delta F: {mean_delta_fs}")
    print(f"Std delta F: {std_delta_fs}")
    print(f"Mean delta F last: {mean_delta_fs[-1]}")
    print(f"Std delta F last: {std_delta_fs[-1]}")
    
    # Compute reference Delta F
    print(f"> Computing reference Delta F")
    reference_weights = np.ones_like(reference_cvs)
    reference_cv_thresh = [reference_cvs.min(), (reference_cvs.min() + reference_cvs.max()) / 2, reference_cvs.max()]
    reference_Delta_F = DeltaF_fromweights(
        xi_traj=reference_cvs,
        weights=reference_weights,
        cv_thresh=reference_cv_thresh,
        T=equil_temp,
    )
    
    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    mask = ~np.isnan(mean_delta_fs)
    if np.any(mask):
        ax.plot(
            time_axis[mask], mean_delta_fs[mask], 
            color=blue, linewidth=4,
            zorder=4
        )
        ax.fill_between(
            time_axis[mask], 
            mean_delta_fs[mask] - std_delta_fs[mask],
            mean_delta_fs[mask] + std_delta_fs[mask],
            alpha=0.2, color=blue, linewidth=1,
        )
    for idx, delta_f in enumerate(all_delta_fs):
        mask = ~np.isnan(delta_f)
        if np.any(mask):
            ax.plot(
                all_times[idx][mask], delta_f[mask],
                color=blue, linewidth=2, alpha=0.2,
                label=f"OPES {idx}",
                zorder=2
            )
    if reference_Delta_F is not None and not np.isnan(reference_Delta_F):
        ax.axhline(
            y=reference_Delta_F, color=COLORS[1], linestyle='--', 
            label='Reference', linewidth=4,
            zorder=6
        )
        ax.fill_between(
            [0, time_axis[-1]], reference_Delta_F - 4, reference_Delta_F + 4,
            color=COLORS[1], alpha=0.2,
        )
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    if cfg.molecule == "cln025":
        ax.set_yticks([-15, 0, 15, 30])
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.set_xlim(0.0, time_axis[-1])
    if cfg.molecule == "cln025":
        ax.set_ylim(-20, 35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=FONTSIZE_SMALL)
    plt.yticks(fontsize=FONTSIZE_SMALL)
    plt.xlabel('Time [ns]', fontsize=FONTSIZE_SMALL)
    plt.ylabel(r'$\Delta F$ [kJ/mol]', fontsize=FONTSIZE_SMALL)
    format_plot_axes(
        ax, fig=fig, 
        model_type=cfg.method, 
        show_y_labels=True,
        align_ylabels=True
    )
    
    # Logging
    save_plot_dual_format(str(analysis_dir), filename, dpi=300, bbox_inches="tight")
    logger.info(f"Free energy curve saved to {analysis_dir}")
    log_info = {
        "free_energy_curve": wandb.Image(str(analysis_dir / f"{filename}.png")),
        "free_energy_difference": round(mean_delta_fs[-1], 2),
        "free_energy_difference_std": round(std_delta_fs[-1], 2),
        "free_energy_difference_reference": round(reference_Delta_F, 2),
        "free_energy_difference_mae": round(np.abs(mean_delta_fs[-1] - reference_Delta_F), 2)
    }
    wandb.log(log_info)
    logger.info(pformat(log_info, indent=4, width=120))
    plt.close()
    
    return


def plot_rmsd_analysis(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
):
    """Calculate and plot alpha carbon RMSD to reference PDB"""
    logger.info("Calculating alpha carbon RMSD to reference structure...")
    
    pbar = tqdm(
        range(max_seed + 1),
        desc="Computing RMSD"
    )
    for seed in pbar:
        filename = f"rmsd_analysis_{seed}"
        # if check_image_exists(str(analysis_dir), filename):
        #     print(f"✓ RMSD analysis plot already exists: {filename}")
        #     wandb.log({
        #         f"rsmd/{seed}": wandb.Image(str(analysis_dir / f"{filename}.png")),
        #     })
        #     continue
        
        # else:
        try:
            ref_pdb_path = f"./data/{cfg.molecule.upper()}/folded.pdb"
            ref_traj = md.load_pdb(ref_pdb_path)
        
            # Load trajectory and compute RMSD
            traj_file = log_dir / "analysis" / f"{seed}_tc.xtc"
            try:
                traj = md.load_xtc(
                    traj_file,
                    top=ref_pdb_path,
                )
                traj.center_coordinates()
                rmsd_values = md.rmsd(
                    traj,
                    ref_traj,
                    atom_indices = traj.topology.select("name CA")                        
                )
            except Exception as e:
                logger.warning(f"Error processing trajectory for seed {seed}: {e}")
                continue
        
            # Load COLVAR data
            colvar_file = log_dir / f"{seed}" / "COLVAR"
            if not colvar_file.exists():
                logger.warning(f"COLVAR file not found: {colvar_file}")
                continue
            traj_dat = np.genfromtxt(colvar_file, skip_header=1)
            time = traj_dat[:, 0]
            final_time = time[-1] / 1000
            time_grid = np.linspace(0, final_time, num=len(rmsd_values))
            
            # Plot RMSD over time
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            plt.plot(time_grid, rmsd_values, color=blue, linewidth=4, label='RMSD')
            ax.set_xlabel('Time (ns)', fontsize=FONTSIZE_SMALL)
            ax.set_ylabel('RMSD (nm)', fontsize=FONTSIZE_SMALL)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
            format_plot_axes(
                ax, fig=fig, 
                model_type=cfg.method, 
                show_y_labels=(cfg.method == "tda"),
                align_ylabels=True
            )
            
            # Save plot
            save_plot_dual_format(str(analysis_dir), filename, dpi=300, bbox_inches="tight")
            logger.info(f"RMSD analysis saved to {analysis_dir}")
            wandb.log({
                f"rmsd/{seed}": wandb.Image(str(analysis_dir / f"{filename}.png")),
                "max_rmsd": round(float(np.max(rmsd_values)), 2),
                "min_rmsd": round(float(np.min(rmsd_values)), 2)
            })
            
            plt.close()
    
        except Exception as e:
            logger.error(f"Error in RMSD analysis: {e}")
            raise


def plot_tica_scatter(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
):
    """Plot TICA scatter with simulation trajectory overlay"""
    logger.info("Creating TICA scatter plot...")
    
    # Process simulation trajectory
    pbar = tqdm(
        range(max_seed + 1),
        desc="Computing TICA scatter"
    )
    for seed in pbar:
        filename = f"tica_scatter_{seed}"
        # if check_image_exists(str(analysis_dir), filename):
        #     print(f"✓ TICA scatter plot already exists: {filename}")
        #     wandb.log({
        #         f"tica/{seed}": wandb.Image(str(analysis_dir / f"{filename}.png"))
        #     })
        #     continue
    
        # else:
        try:
            lag = 1000 if cfg.molecule == "1fme" else 10
            # Load TICA model
            if cfg.molecule == "cln025":
                tica_model_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_model_switch_lag{lag}.pkl"
            else:
                tica_model_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_model_lag{lag}.pkl"
            with open(tica_model_path, 'rb') as f:
                tica_model = pickle.load(f)
            
            # Load coordinates for background
            tica_coord_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_coord_lag{lag}.npy"
            if Path(tica_coord_path).exists():
                tica_coord_full = np.load(tica_coord_path)
            else:
                cad_full_path = f"./data/{cfg.molecule.upper()}/cad-switch.pt" if cfg.molecule == "cln025" else f"./data/{cfg.molecule.upper()}/cad.pt"
                cad_full = torch.load(cad_full_path).numpy()
                tica_coord_full = tica_model.transform(cad_full)
        
            # Load trajectory data
            traj_file = analysis_dir / f"{seed}_tc.xtc"
            if not traj_file.exists()   :
                logger.warning(f"No trajectory file found for seed {seed}")
                continue

            # top_file = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_from_mae.pdb"
            top_file = f"./data/{cfg.molecule.upper()}/folded.gro"
            if not Path(top_file).exists():
                logger.warning(f"Topology file not found: {top_file}")
                continue
            try:
                traj = md.load(str(traj_file), top=top_file)
                ca_resid_pair = np.array([(a.index, b.index) for a, b in combinations(list(traj.topology.residues), 2)])
                ca_pair_contacts, _ = md.compute_contacts(traj, scheme="ca", contacts=ca_resid_pair, periodic=False)
                if cfg.molecule == "cln025":
                    ca_pair_contacts_switch = (1 - np.power(ca_pair_contacts / 0.8, 6)) / (1 - np.power(ca_pair_contacts / 0.8, 12))
                    tica_coord = tica_model.transform(ca_pair_contacts_switch)
                else:
                    tica_coord = tica_model.transform(ca_pair_contacts)
            except Exception as e:
                logger.warning(f"Error processing trajectory for seed {seed}: {e}")
                continue
        
            # Create plot
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            h = ax.hist2d(
                tica_coord_full[:, 0], tica_coord_full[:, 1], 
                bins=100, norm=LogNorm(), alpha=0.3
            )
            # plt.colorbar(h[3], ax=ax, label='Log Density')
            ax.scatter(
                tica_coord[:, 0], tica_coord[:, 1], 
                color=blue, s=2, alpha=0.5,
            )
            ax.set_xlabel("TIC 1", fontsize=FONTSIZE_SMALL)
            ax.set_ylabel("TIC 2", fontsize=FONTSIZE_SMALL)
            format_plot_axes(
                ax, fig=fig, 
                model_type=cfg.method, 
                show_y_labels=(cfg.method == "tda"),
                align_ylabels=True
            )
            save_plot_dual_format(str(analysis_dir), filename, dpi=300, bbox_inches="tight")
            logger.info(f"TICA scatter plot saved to {analysis_dir}")
            wandb.log({
                f"tica/{seed}": wandb.Image(str(analysis_dir / f"{filename}.png"))
            })
            plt.close()

        except Exception as e:
            logger.error(f"Error in TICA scatter analysis: {e}")
            raise    


def plot_cv_over_time(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
):
    """Plot CV values over time for simulation trajectories"""
    logger.info("Creating CV over time plots...")
    
    pbar = tqdm(
        range(max_seed + 1),
        desc="Computing CV over time"
    )
    for seed in pbar:
        filename_time = f"cv_over_time_{seed}"
        filename_histogram = f"cv_histogram_{seed}"
        # if check_image_exists(str(analysis_dir), filename_time) and check_image_exists(str(analysis_dir), filename_histogram):
        #     print(f"✓ CV over time and histogram plots already exist: {filename_time} and {filename_histogram}")
        #     continue
        
        # else:
        try:
            colvar_file = log_dir / f"{seed}" / "COLVAR"
            if not colvar_file.exists():
                logger.warning(f"COLVAR file not found: {colvar_file}")
                continue
            traj_dat = np.genfromtxt(colvar_file, skip_header=1)
            time = traj_dat[:, 0] / 1000
            cv = traj_dat[:, 1]
            print(f"Time shape: {time.shape}")
            print(f"CV shape: {cv.shape}")
            print(f"Time last: {time[-1]}")
            print(f"CV last: {cv[-1]}")
            
            # Plot - CV over time
            # if not check_image_exists(str(analysis_dir), filename_time):
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.plot(time, cv, label=f"CV", alpha=0.8, linewidth=4, color=blue)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xlabel("Time (ns)", fontsize=FONTSIZE_SMALL)
            ax.set_ylabel("CV Values", fontsize=FONTSIZE_SMALL)
            format_plot_axes(
                ax, fig=fig, 
                model_type=cfg.method, 
                show_y_labels=(cfg.method == "tda"),
                align_ylabels=True
            )
            save_plot_dual_format(str(analysis_dir), filename_time, dpi=300, bbox_inches="tight")
            logger.info(f"CV over time plot saved to {analysis_dir}")
            wandb.log({
                f"cv_over_time/{seed}": wandb.Image(str(analysis_dir / f"{filename_time}.png"))
            })
            plt.close()
            
            # Plot - CV histogram
            # if not check_image_exists(str(analysis_dir), filename_histogram):
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.hist(cv, bins=50, alpha=0.7, color=blue, edgecolor='black', log=True)
            ax.set_xlabel("CV Values", fontsize=FONTSIZE_SMALL)
            ax.set_ylabel("Frequency", fontsize=FONTSIZE_SMALL)
            format_plot_axes(
                ax, fig=fig, 
                model_type=cfg.method, 
                show_y_labels=(cfg.method == "tda"),
                align_ylabels=True
            )
            save_plot_dual_format(str(analysis_dir), filename_histogram, dpi=300, bbox_inches="tight")
            logger.info(f"CV histogram plot saved to {analysis_dir}")
            wandb.log({
                f"cv_histogram/{seed}": wandb.Image(str(analysis_dir / f"{filename_histogram}.png"))
            })
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in CV over time analysis: {e}")
            raise


@hydra.main(
    config_path="config",
    config_name="tda",
    version_base=None,
)
def main(cfg):
    """Main function using Hydra configuration"""
    logger.info("Starting OPES analysis with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup paths
    base_simulation_dir = Path(f"{os.getcwd()}/simulations") / cfg.molecule / cfg.method
    data_dir = Path(f"{os.getcwd()}/data") / cfg.molecule.upper()
    log_dir = base_simulation_dir / cfg.date
    analysis_dir = log_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    max_seed = cfg.opes.max_seed
    sigma = cfg.opes.sigma
    
    # Initialize wandb
    config = OmegaConf.to_container(cfg)
    wandb.init(
        project="opes-analysis",
        entity="eddy26",
        tags=cfg.tags if hasattr(cfg, 'tags') else ['debug'],
        config=config,
    )
    
    try:
        # logger.info("Post processing trajectory...")
        if cfg.analysis.gmx:
            gmx_process_trajectory(cfg, data_dir, log_dir, analysis_dir, max_seed)
            gmx_process_energy(log_dir, analysis_dir, max_seed)
        else:
            logger.info("Skipping gmx trajectory post-processing and energy calculation...")
        
        # Run analysis functions
        logger.info("Running CV analysis...")
        plot_cv_over_time(cfg, log_dir, max_seed, analysis_dir)
        
        logger.info("Running RMSD analysis...")
        plot_rmsd_analysis(cfg, log_dir, max_seed, analysis_dir)
        
        logger.info("Running TICA scatter analysis...")
        plot_tica_scatter(cfg, log_dir, max_seed, analysis_dir)
        
        logger.info("Running free energy analysis...")
        reference_cvs = compute_cv_values(cfg, max_seed, batch_size=10000)
        plot_free_energy_curve(cfg, log_dir, max_seed, analysis_dir, reference_cvs)
        plot_pmf(cfg, sigma, log_dir, max_seed, analysis_dir, reference_cvs)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise
    
    finally:
        wandb.finish()




if __name__ == "__main__":
    main()