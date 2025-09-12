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
        if os.path.exists(trj_save_path):
            print(f"✓ trjconv file already exists: {trj_save_path}")
            continue
        
        else:
            cmd = [
                "gmx", "trjconv",
                "-f", f"{log_dir}/{seed}.xtc",
                "-s", f"{data_dir}/nvt_0.tpr",
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
        if os.path.exists(edr_save_path):
            print(f"✓ energy file already exists: {edr_save_path}")
            continue
        
        else:
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
    equil_temp = 340
    plot_path = analysis_dir / "pmf.png"
    if os.path.exists(plot_path):
        print(f"✓ PMF plot already exists: {plot_path}")
        wandb.log({
            "pmf": wandb.Image(str(plot_path)),
        })
        return
    
    else:
        print(f"> Computing PMF")
        
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
            cv_grid, reference_pmf,
            color=COLORS[1], linewidth=4, linestyle="--",
            label="Reference",
        )
        ax.plot(
            cv_grid, mean_pmf,
            color=blue, linewidth=4,
        )
        ax.fill_between(
            cv_grid, mean_pmf - std_pmf, mean_pmf + std_pmf,
            alpha=0.2, color=blue, linewidth=1
        )
        for idx, pmf in enumerate(all_pmfs):
            ax.plot(
                cv_grid, pmf,
                color=blue, linewidth=2, alpha=0.2,
                label=f"OPES {idx}"
            )
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=FONTSIZE_SMALL)
        ax.set_xlabel("CV", fontsize=FONTSIZE_SMALL)
        ax.set_ylabel("PMF [kJ/mol]", fontsize=FONTSIZE_SMALL)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"PMF plot saved to {plot_path}")
        wandb.log({
            "pmf": wandb.Image(str(plot_path)),
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
    plot_path = analysis_dir / "free_energy_curve.png"
    if os.path.exists(plot_path):
        print(f"✓ Free energy curve plot already exists: {plot_path}")
        wandb.log({
            "free_energy_curve": wandb.Image(str(plot_path)),
        })
        return
    
    else:
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
                all_times.append(time[step_grid] * 0.001)
                
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
        padded = np.full((len(all_delta_fs), max_len), np.nan, dtype=float)
        for i, x in enumerate(all_delta_fs):
            padded[i, :len(x)] = x
        idx_longest = int(np.argmax([len(t) for t in all_times]))
        time_axis = all_times[idx_longest]
        all_delta_fs = padded
        
        # Compute mean/std ignoring NaNs, but only keep columns where at least one value is present
        has_data = (~np.isnan(all_delta_fs)) & (~np.isinf(all_delta_fs))
        valid_values = padded * has_data.astype(float)
        mean_delta_fs = np.nanmean(valid_values, axis=0)
        std_delta_fs  = np.nanstd(valid_values,  axis=0)
        
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
        if reference_Delta_F is not None and not np.isnan(reference_Delta_F):
            ax.axhline(
                y=reference_Delta_F, color=COLORS[1], linestyle='--', 
                label='Reference', linewidth=4
            )
            ax.fill_between(
                [0, time_axis[-1]], reference_Delta_F - 4, reference_Delta_F + 4,
                color=COLORS[1], alpha=0.2
            )
        mask = ~np.isnan(mean_delta_fs)
        if np.any(mask):
            ax.plot(
                time_axis[mask], mean_delta_fs[mask], 
                color=blue, linewidth=4
            )
            ax.fill_between(
                time_axis[mask], 
                mean_delta_fs[mask] - std_delta_fs[mask],
                mean_delta_fs[mask] + std_delta_fs[mask],
                alpha=0.2, color=blue, linewidth=1
            )
        for idx, delta_f in enumerate(all_delta_fs):
            mask = ~np.isnan(delta_f)
            if np.any(mask):
                ax.plot(
                    time_axis[mask], delta_f[mask],
                    color=blue, linewidth=2, alpha=0.2,
                    label=f"OPES {idx}"
                )
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        if cfg.molecule == "cln025":
            ax.set_yticks([-10, 0, 10, 20])
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.set_xlim(0.0, time_axis[-1])
        if cfg.molecule == "cln025":
            ax.set_ylim(-20, 30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=FONTSIZE_SMALL)
        plt.yticks(fontsize=FONTSIZE_SMALL)
        plt.xlabel('Time [ns]', fontsize=FONTSIZE_SMALL)
        plt.ylabel(r'$\Delta F$ [kJ/mol]', fontsize=FONTSIZE_SMALL)
        # plt.title(f'Free Energy Difference (CVs) - {cfg.method}')
        # plt.legend(fontsize=FONTSIZE_SMALL)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Logging
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Free energy curve saved to {plot_path}")
        log_info = {
            "free_energy_curve": wandb.Image(str(plot_path)),
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
        plot_path = analysis_dir / f"rmsd_analysis_{seed}.png"
        if os.path.exists(plot_path):
            print(f"✓ RMSD analysis plot already exists: {plot_path}")
            wandb.log({
                "rmsd_analysis": wandb.Image(str(plot_path)),
            })
            continue
        
        else:
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
                plt.xlabel('Time (ns)')
                plt.ylabel('RMSD (nm)')
                ax.set_title(f'CA RMSD to Reference - {cfg.method}, {seed}')
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
                # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                logger.info(f"RMSD analysis saved to {plot_path}")
                wandb.log({
                    "rmsd_analysis": wandb.Image(str(plot_path)),
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
        plot_path = analysis_dir / f"tica_scatter_{seed}.png"
        if os.path.exists(plot_path):
            print(f"✓ TICA scatter plot already exists: {plot_path}")
            wandb.log({"tica_scatter": wandb.Image(str(plot_path))})
            continue
    
        else:
            try:
                # Load TICA model
                if cfg.molecule == "cln025":
                    tica_model_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_model_switch_lag10.pkl"
                else:
                    tica_model_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_model_lag10.pkl"
                with open(tica_model_path, 'rb') as f:
                    tica_model = pickle.load(f)
                
                # Load coordinates for background
                tica_coord_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_coord_lag10.npy"
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
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                h = ax.hist2d(
                    tica_coord_full[:, 0], tica_coord_full[:, 1], 
                    bins=100, norm=LogNorm(), alpha=0.3
                )
                plt.colorbar(h[3], ax=ax, label='Log Density')
                ax.scatter(
                    tica_coord[:, 0], tica_coord[:, 1], 
                    color=blue, s=2, alpha=0.5,
                )
                ax.set_xlabel("TIC 1")
                ax.set_ylabel("TIC 2")
                ax.set_title(f'TICA Scatter Plot - {cfg.method}')
                ax.grid(True, alpha=0.3)
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                logger.info(f"TICA scatter plot saved to {plot_path}")
                wandb.log({"tica_scatter": wandb.Image(str(plot_path))})
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
        plot_path = analysis_dir / f"cv_over_time_{seed}.png"
        plot_path_histogram = analysis_dir / f"cv_histogram_{seed}.png"
        if os.path.exists(plot_path) and os.path.exists(plot_path_histogram):
            print(f"✓ CV over time and histogram plots already exist: {plot_path} and {plot_path_histogram}")
            continue
        
        else:
            try:
                colvar_file = log_dir / f"{seed}" / "COLVAR"
                if not colvar_file.exists():
                    logger.warning(f"COLVAR file not found: {colvar_file}")
                    continue
                traj_dat = np.genfromtxt(colvar_file, skip_header=1)
                time = traj_dat[:, 0] / 1000
                cv = traj_dat[:, 1]
                
                # Plot - CV over time
                if not os.path.exists(plot_path):
                    fig = plt.figure(figsize=(5, 3))
                    ax = fig.add_subplot(111)
                    ax.plot(time, cv, label=f"CV", alpha=0.8, linewidth=4, color=blue)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
                    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                    ax.set_xlabel("Time (ns)")
                    ax.set_ylabel("CV Values")
                    ax.set_title(f"CV Evolution - {cfg.method} {seed}")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    logger.info(f"CV over time plot saved to {plot_path}")
                    wandb.log({
                        f"cv_over_time_{seed}": wandb.Image(str(plot_path))
                    })
                    plt.close()
                
                # Plot - CV histogram
                if not os.path.exists(plot_path_histogram):
                    fig = plt.figure(figsize=(5, 3))
                    ax = fig.add_subplot(111)
                    ax.hist(cv, bins=50, alpha=0.7, color=blue, edgecolor='black', log=True)
                    ax.set_xlabel("CV Values")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"CV Histogram - {cfg.method} {seed}")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    logger.info(f"CV histogram plot saved to {plot_path}")
                    wandb.log({
                        f"cv_histogram_{seed}": wandb.Image(str(plot_path))
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