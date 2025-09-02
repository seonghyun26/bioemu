import os
import sys
import torch
import hydra
import argparse
import subprocess
import pickle
import wandb
import logging

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md

from tqdm import tqdm
from matplotlib.colors import LogNorm
from itertools import combinations
from pathlib import Path
from omegaconf import OmegaConf

from adaptive_sampling.processing_tools import mbar
from adaptive_sampling.processing_tools.utils import DeltaF_fromweights

# Add the parent directory to Python path to import analysis functions
from src import *
from src.constant import COLORS, FONTSIZE_SMALL

# Set up logging
logger = logging.getLogger(__name__)
blue = (70 / 255, 110 / 255, 250 / 255)
R = 0.008314462618  # kJ/mol/K

def label_by_hbond(
    traj,
    distance_cutoff=0.35,
    angle_cutoff=110,
    bond_number_cutoff=3
):
    """
    Generate binary labels for folded/unfolded states based on at least 3 bonds among eight bonds
    - TYR1N-TYR10O (backbone)
    - TYR1N-TYR10OXT (backbone)
    - ASP3N-TYR8O
    - THR6OG1-ASP3O
    - THR6N-ASP3OD1
    - THR6N-ASP3OD2
    - GLY7N-ASP3O
    - TYR10N-TYR1O

    Args:
        traj (mdtraj): mdtraj trajectory object
        distance_cutoff (float): donor-acceptor distance cutoff in nm (default 0.35 nm = 3.5 angstrom)
        angle_cutoff (float): hydrogen bond angle cutoff in degrees (default 110 deg)
        bond_number_cutoff (int): minimum number of bonds to be considered as folded (default 3)

    Returns:
        labels (np.array): binary array (1: folded, 0: unfolded)
        bond_sum (np.array): total number of bonds per frame
        distances (np.array): distance matrix for all bonds
    """
    distances = []
    
    # TYR1N-TYR10O (backbone)
    logger.info("Bond: TYR1N-TYR10O (1/8)")
    donor_idx = traj.topology.select('residue 1 and name N')[0]  # Tyr1:N
    acceptor_idx = traj.topology.select('residue 10 and name O')[0]  # Tyr10:O
    distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
    
    # Try to find hydrogen atoms attached to TYR1N
    try:
        hydrogen_indices = traj.topology.select('residue 1 and name H')
        if len(hydrogen_indices) == 0:
            # If no H found, try by index (common in GROMACS)
            hydrogen_indices = [1, 2, 3]
        
        bond_found = False
        for h_idx in hydrogen_indices:
            try:
                angle = md.compute_angles(traj, [[donor_idx, h_idx, acceptor_idx]]) * (180.0 / np.pi)
                label = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
                if np.any(label):
                    bond_found = True
                    break
            except:
                continue
        
        if not bond_found:
            label = np.zeros(traj.n_frames, dtype=int)
    except:
        label = np.zeros(traj.n_frames, dtype=int)
    
    label_TYR1N_TYR10O = label
    distances.append(distance[:, 0])

    # TYR1N-TYR10OXT (backbone)
    logger.info("Bond: TYR1N-TYR10OXT (2/8)")
    try:
        acceptor_idx = traj.topology.select('residue 10 and name OXT')[0]  # Tyr10:OXT
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        bond_found = False
        for h_idx in hydrogen_indices:
            try:
                angle = md.compute_angles(traj, [[donor_idx, h_idx, acceptor_idx]]) * (180.0 / np.pi)
                label = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
                if np.any(label):
                    bond_found = True
                    break
            except:
                continue
        
        if not bond_found:
            label = np.zeros(traj.n_frames, dtype=int)
        label_TYR1N_TYR10OXT = label
        distances.append(distance[:, 0])
    except:
        label_TYR1N_TYR10OXT = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # ASP3N-TYR8O
    logger.info("Bond: ASP3N-TYR8O (3/8)")
    try:
        donor_idx = traj.topology.select('residue 3 and name N')[0]
        acceptor_idx = traj.topology.select('residue 8 and name O')[0]
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        # Find hydrogen attached to ASP3N
        hydrogen_indices = traj.topology.select('residue 3 and name H')
        if len(hydrogen_indices) > 0:
            hydrogen_idx = hydrogen_indices[0]
            angle = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]]) * (180.0 / np.pi)
            labels_ASP3N_TYR8O = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
        else:
            labels_ASP3N_TYR8O = np.zeros(traj.n_frames, dtype=int)
        distances.append(distance[:, 0])
    except:
        labels_ASP3N_TYR8O = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # THR6OG1-ASP3O
    logger.info("Bond: THR6OG1-ASP3O (4/8)")
    try:
        donor_idx = traj.topology.select('residue 6 and name OG1')[0]
        acceptor_idx = traj.topology.select('residue 3 and name O')[0]
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        hydrogen_indices = traj.topology.select('residue 6 and name HG1')
        if len(hydrogen_indices) > 0:
            hydrogen_idx = hydrogen_indices[0]
            angle = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]]) * (180.0 / np.pi)
            labels_THR6OG1_ASP3O = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
        else:
            labels_THR6OG1_ASP3O = np.zeros(traj.n_frames, dtype=int)
        distances.append(distance[:, 0])
    except:
        labels_THR6OG1_ASP3O = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # THR6N-ASP3OD1
    logger.info("Bond: THR6N-ASP3OD1 (5/8)")
    try:
        donor_idx = traj.topology.select('residue 6 and name N')[0]
        acceptor_idx = traj.topology.select('residue 3 and name OD1')[0]
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        hydrogen_indices = traj.topology.select('residue 6 and name H')
        if len(hydrogen_indices) > 0:
            hydrogen_idx = hydrogen_indices[0]
            angle = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]]) * (180.0 / np.pi)
            labels_THR6N_ASP3OD1 = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
        else:
            labels_THR6N_ASP3OD1 = np.zeros(traj.n_frames, dtype=int)
        distances.append(distance[:, 0])
    except:
        labels_THR6N_ASP3OD1 = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # THR6N-ASP3OD2
    logger.info("Bond: THR6N-ASP3OD2 (6/8)")
    try:
        donor_idx = traj.topology.select('residue 6 and name N')[0]
        acceptor_idx = traj.topology.select('residue 3 and name OD2')[0]
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        hydrogen_indices = traj.topology.select('residue 6 and name H')
        if len(hydrogen_indices) > 0:
            hydrogen_idx = hydrogen_indices[0]
            angle = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]]) * (180.0 / np.pi)
            labels_THR6N_ASP3OD2 = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
        else:
            labels_THR6N_ASP3OD2 = np.zeros(traj.n_frames, dtype=int)
        distances.append(distance[:, 0])
    except:
        labels_THR6N_ASP3OD2 = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # GLY7N-ASP3O
    logger.info("Bond: GLY7N-ASP3O (7/8)")
    try:
        donor_idx = traj.topology.select('residue 7 and name N')[0]
        acceptor_idx = traj.topology.select('residue 3 and name O')[0]
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        hydrogen_indices = traj.topology.select('residue 7 and name H')
        if len(hydrogen_indices) > 0:
            hydrogen_idx = hydrogen_indices[0]
            angle = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]]) * (180.0 / np.pi)
            labels_GLY7N_ASP3O = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
        else:
            labels_GLY7N_ASP3O = np.zeros(traj.n_frames, dtype=int)
        distances.append(distance[:, 0])
    except:
        labels_GLY7N_ASP3O = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # TYR10N-TYR1O
    logger.info("Bond: TYR10N-TYR1O (8/8)")
    try:
        donor_idx = traj.topology.select('residue 10 and name N')[0]
        acceptor_idx = traj.topology.select('residue 1 and name O')[0]
        distance = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
        
        hydrogen_indices = traj.topology.select('residue 10 and name H')
        if len(hydrogen_indices) > 0:
            hydrogen_idx = hydrogen_indices[0]
            angle = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]]) * (180.0 / np.pi)
            labels_TYR10N_TYR1O = ((distance[:,0] < distance_cutoff) & (angle[:,0] > angle_cutoff)).astype(int)
        else:
            labels_TYR10N_TYR1O = np.zeros(traj.n_frames, dtype=int)
        distances.append(distance[:, 0])
    except:
        labels_TYR10N_TYR1O = np.zeros(traj.n_frames, dtype=int)
        distances.append(np.full(traj.n_frames, np.nan))

    # All bonds
    bond_sum = (label_TYR1N_TYR10O + label_TYR1N_TYR10OXT + labels_ASP3N_TYR8O + 
                labels_THR6OG1_ASP3O + labels_THR6N_ASP3OD1 + labels_THR6N_ASP3OD2 + 
                labels_GLY7N_ASP3O + labels_TYR10N_TYR1O)
    labels = bond_sum >= bond_number_cutoff
    distances = np.array(distances)

    logger.info(f"Folded frames: {np.sum(labels)} / {len(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
    
    return labels, bond_sum, distances


def post_process_trajectory(
    log_dir: Path,
    analysis_dir: Path,
    seed: int
):
    """Post process trajectory."""
    
    # GROMACS command for post procesing trajectory with trjconv
    cmd = [
        "gmx", "trjconv",
        "-f", f"{log_dir}/{seed}.xtc",
        "-pbc", "nojump",
        "-o", f"{analysis_dir}/{seed}_tc.xtc",
    ]
    
    # Run and wait for completion
    print(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if process.returncode == 0:
            print("✓ gmx trjconv completed successfully")
            print(f"Created trjconv file: {analysis_dir}/{seed}_tc.xtc")
        else:
            print(f"✗ gmx trjconv failed with return code {process.returncode}")
        if process.stdout:
            print("STDOUT:", process.stdout)
        if process.stderr:
            print("GROMACSOUT:", process.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"gmx trjconv failed: {e}")
    
    
def compute_ref_delta_f(
    cfg,
):
    """Compute reference free energy difference using full trajectories.

    Steps:
    - Load each seed's trajectory and corresponding COLVAR bias
    - Label frames as folded/unfolded via hydrogen-bond criteria (for CLN025)
    - Reweight using exp(beta * bias) and aggregate across seeds
    -  F = kT * ln(P_folded / P_unfolded)
    """
    try:
        base_simulation_dir = Path(f"{os.getcwd()}/simulations") / cfg.molecule / cfg.method
        log_dir = base_simulation_dir / cfg.date

        equil_temp = 340  # K (consistent with analysis)
        kT = R * equil_temp
        beta = 1.0 / kT
        total_weight_folded = 0.0
        total_weight_unfolded = 0.0


    except Exception as e:
        logger.error(f"Failed to compute reference  F: {e}")
        return float('nan')



def compute_energy(
    log_dir: Path,
    analysis_dir: Path,
    seed: int,
):
    """Compute energy from trajectory data."""
    
    # GROMACS command for energy calculation
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
            print("✓ gmx energy completed successfully")
            print(f"Created energy file: {analysis_dir}/{seed}.xvg")
        else:
            print(f"✗ gmx energy failed with return code {process.returncode}")
        if process.stdout:
            print("STDOUT:", process.stdout)
        if process.stderr:
            print("GROMACSOUT:", process.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"gmx energy failed: {e}")


def plot_free_energy_curve_by_bond(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path
):
    """Plot free energy curve from OPES simulation data using folded/unfolded state analysis"""
    logger.info("Plotting free energy curve using hydrogen bond-based folded/unfolded states...")
    
    # NOTE: modify this function
    # try:
    # Collect data from all seeds
    all_delta_fs = []
    all_times = []
    
    for seed in range(max_seed + 1):
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        if not colvar_file.exists():
            logger.warning(f"COLVAR file not found: {colvar_file}")
            continue
        
        try:
            # Load COLVAR data
            colvar_data = np.loadtxt(colvar_file, comments='#')
            if len(colvar_data) == 0:
                continue                
            with open(colvar_file, 'r') as f:
                header = f.readline().strip()
                keys = header.split()[2:]  # Skip '#!' and 'FIELDS'
            time_idx = keys.index('time')
            times = colvar_data[:, time_idx] / 1000  # Convert ps to ns
            
            # Load trajectory data
            traj_file = log_dir / f"{seed}.xtc"
            top_file = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_from_mae.pdb"
            if not traj_file:
                logger.warning(f"No trajectory file found for seed {seed}")
                continue
            if not Path(top_file).exists():
                logger.warning(f"Topology file not found: {top_file}")
                continue
            traj = md.load(str(traj_file), top=top_file)
            
            # Ensure time arrays match
            n_frames = min(len(times), traj.n_frames)
            times = times[:n_frames]
            traj = traj[:n_frames]
            
            # Calculate hydrogen bond labels
            if cfg.molecule.lower() == "cln025":
                labels, bond_sum, distances = label_by_hbond(traj)
                
                # Calculate delta F over time windows
                window_size = max(1000, len(labels) // 20)  # Adaptive window size
                delta_fs = []
                time_points = []
                
                for i in range(window_size, len(labels), window_size):
                    window_labels = labels[max(0, i-window_size):i]
                    window_time = times[i-1]  # Time at end of window
                    
                    # Count folded and unfolded states
                    folded_count = np.sum(window_labels == 1)
                    unfolded_count = np.sum(window_labels == 0)
                    
                    if folded_count > 0 and unfolded_count > 0:
                        # Free energy difference: -kT * ln(N_unfolded/N_folded)
                        kT = 2.49  # kJ/mol at 300K
                        delta_f = -kT * np.log(unfolded_count / folded_count)
                        delta_fs.append(delta_f)
                        time_points.append(window_time)
                    elif folded_count > 0:  # Only folded states
                        delta_fs.append(0.0)  # Folded is reference
                        time_points.append(window_time)
                    elif unfolded_count > 0:  # Only unfolded states
                        delta_fs.append(15.0)  # Large positive value
                        time_points.append(window_time)
                    else:
                        delta_fs.append(np.nan)
                        time_points.append(window_time)
                
                all_delta_fs.append(delta_fs)
                all_times.append(time_points)
            else:
                logger.warning(f"Hydrogen bond analysis not implemented for molecule: {cfg.molecule}")
                continue
            
        except Exception as e:
            logger.warning(f"Error processing data for seed {seed}: {e}")
            continue
    
    if not all_delta_fs:
        logger.warning("No valid data found for free energy analysis")
        return
    
    # Find common time range
    min_length = min(len(df) for df in all_delta_fs)
    if min_length == 0:
        logger.warning("All delta_f arrays are empty")
        return
    
    # Truncate all arrays to same length
    truncated_delta_fs = [df[:min_length] for df in all_delta_fs]
    truncated_times = [times[:min_length] for times in all_times]
    
    # Convert to arrays and compute statistics
    delta_fs_array = np.array(truncated_delta_fs)
    mean_delta_fs = np.nanmean(delta_fs_array, axis=0)
    std_delta_fs = np.nanstd(delta_fs_array, axis=0)
    
    # Use time from first seed (should be similar for all)
    time_axis = truncated_times[0] if truncated_times else np.arange(min_length)
    
    # Plot
    plt.figure(figsize=(5, 3))
    mask = ~np.isnan(mean_delta_fs)
    if np.any(mask):
        plt.plot(time_axis[mask], mean_delta_fs[mask], 
                color=COLORS[0], linewidth=2, label='Simulation')
        plt.fill_between(time_axis[mask], 
                        mean_delta_fs[mask] - std_delta_fs[mask],
                        mean_delta_fs[mask] + std_delta_fs[mask],
                        alpha=0.3, color=COLORS[0])
    
    # Reference line (if known)
    ref_delta_f = 10.06  # kJ/mol - known reference for CLN025
    plt.axhline(y=ref_delta_f, color=COLORS[1], linestyle='--', 
                label='Reference', linewidth=2)
    plt.fill_between(time_axis, ref_delta_f - 0.5, ref_delta_f + 0.5,
                    color=COLORS[1], alpha=0.2)
    
    plt.xlabel('Time (ns)')
    plt.ylabel(r'$\Delta F$ (kJ/mol)')
    plt.title(f'Free Energy Difference (Folded vs Unfolded) - {cfg.method}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = analysis_dir / "free_energy_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Free energy curve saved to {plot_path}")
    
    # Log to wandb
    wandb.log({
        "free_energy_curve": wandb.Image(str(plot_path)),
        "final_delta_f_mean": float(mean_delta_fs[mask][-1]) if np.any(mask) else np.nan,
        "final_delta_f_std": float(std_delta_fs[mask][-1]) if np.any(mask) else np.nan
    })
    
    plt.close()
        
    # except Exception as e:
    #     logger.error(f"Error in free energy analysis: {e}")
    #     raise


def plot_pmf(
    cfg,
    sigma: float,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
):
    equil_temp = 340
    all_pmfs = []
    
    print(f"Plotting PMF for {cfg.method}, sigma={sigma}, seed={max_seed}")
    for seed in range(max_seed + 1):
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        if not colvar_file.exists():
            logger.warning(f"COLVAR file not found: {colvar_file}")
            continue
        
        # Load COLVAR data
        colvar_data = np.genfromtxt(colvar_file, skip_header=1)
        cv = colvar_data[:, 1]
        cv_grid = np.arange(cv.min(), cv.max() + sigma / 2, sigma)
        bias = colvar_data[:, 2]
        beta = 1.0 / (R * equil_temp)
        W = np.exp(beta * bias)  
        
        print(cv_grid)
        print(cv)
        print(W)
        mask1 = ~np.isnan(cv)
        mask2 = ~np.isnan(W)
        mask3 = ~np.isnan(cv_grid)
        mask = mask1 & mask2 & mask3
        pmf, _ = mbar.pmf_from_weights(
            cv_grid[mask],
            cv[mask],
            W[mask],
            equil_temp=equil_temp
        )
        all_pmfs.append(pmf)

    all_pmfs = np.array(all_pmfs)
    print(all_pmfs.shape)
    print(all_pmfs)
    all_pmfs -= all_pmfs.min(axis=0)
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(111)
    for pmf in all_pmfs:
        ax.plot(
            cv_grid, pmf,
            color=blue, linewidth=2
        )
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("CV", fontsize=12)
    plt.ylabel("PMF", fontsize=12)
    plt.title(f"PMF - {cfg.method}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_dir / "pmf.png", dpi=300, bbox_inches="tight")
    logger.info(f"PMF plot saved to {analysis_dir}/pmf.png")
    wandb.log({
        "pmf": wandb.Image(str(analysis_dir / "pmf.png"))
    })
    plt.close()
    
    return

def plot_free_energy_curve(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
    ref_delta_f: float = -3.5,
):
    skip_steps = 0
    ns_per_step = 0.004
    unit_steps = 250
    equil_temp = 340
    all_times = []
    all_cvs = []
    all_delta_fs = []
    
    for seed in range(max_seed + 1):
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        if not colvar_file.exists():
            logger.warning(f"COLVAR file not found: {colvar_file}")
            continue
        
        try:
            # Load COLVAR data
            colvar_data = np.genfromtxt(colvar_file, skip_header=1)
            # print(colvar_data)
            time = colvar_data[:, 0]
            cv = colvar_data[:, 1]
            bias = colvar_data[:, 2]
            total_steps = len(colvar_data)
            # print(total_steps)
            step_grid = np.arange(
                skip_steps + unit_steps, total_steps + 1, unit_steps
            )
            all_cvs.append(cv)
            time_axis = step_grid * ns_per_step
            all_times.append(time_axis)
            
            Delta_Fs = []
            beta = 1.0 / (R * equil_temp)
            W = np.exp(beta * bias)  
            cv_grid = np.arange(cv.min(), cv.max() + cfg.sigma / 2, cfg.sigma)
            print(step_grid)
            for current_step in tqdm(step_grid):
                cv_t = cv[skip_steps:current_step]
                W_t = W[skip_steps:current_step]

                Delta_F = DeltaF_fromweights(
                    xi_traj=cv_t,
                    weights=W_t,
                    cv_thresh=[cv.min(), (cv.min() + cv.max()) / 2, cv.max()],
                    T=equil_temp,
                )
                Delta_Fs.append(Delta_F)
            all_delta_fs.append(Delta_Fs)
        
        except Exception as e:
            logger.warning(f"Error processing data for seed {seed}: {e}")
            continue
    
    # Compute mean and std of delta F
    if not all_delta_fs:
        logger.warning("No valid data found for free energy analysis")
        return
    mean_delta_fs = np.mean(all_delta_fs, axis=0)
    std_delta_fs = np.std(all_delta_fs, axis=0)
    time_axis = time_axis
    print(mean_delta_fs)
    
    # Plot
    plt.figure(figsize=(5, 3))
    mask = ~np.isnan(mean_delta_fs)
    if np.any(mask):
        plt.plot(
            time_axis[mask], mean_delta_fs[mask], 
            color=blue, linewidth=2
        )
        plt.fill_between(
            time_axis[mask], 
            mean_delta_fs[mask] - std_delta_fs[mask],
            mean_delta_fs[mask] + std_delta_fs[mask],
            alpha=0.3, color=COLORS[0]
        )
    if ref_delta_f is not None and not np.isnan(ref_delta_f):
        plt.axhline(
            y=ref_delta_f, color=COLORS[1], linestyle='--', 
            label='Reference', linewidth=2
        )
        plt.fill_between(
            time_axis, ref_delta_f - 4, ref_delta_f + 4,
            color=COLORS[1], alpha=0.2
        )
    plt.xlabel('Time (ns)')
    plt.ylabel(r'$\Delta F$ (kJ/mol)')
    plt.title(f'Free Energy Difference (CVs) - {cfg.method}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Logging
    plot_path = analysis_dir / "free_energy_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Free energy curve saved to {plot_path}")
    wandb.log({
        "free_energy_curve": wandb.Image(str(plot_path)),
        "free_energy_difference": mean_delta_fs[-1]
    })
    plt.close()


def plot_rmsd_analysis(
    cfg,
    log_dir: Path,
    max_seed: int,
    analysis_dir: Path,
):
    """Calculate and plot alpha carbon RMSD to reference PDB"""
    logger.info("Calculating alpha carbon RMSD to reference structure...")
    
    try:
        ref_pdb_path = f"./data/{cfg.molecule.upper()}/folded.pdb"
        ref_traj = md.load_pdb(ref_pdb_path)
        
        for seed in range(max_seed + 1):
            traj_file = log_dir / "analysis" / f"{seed}_tc.xtc"

            # Load trajectory and compute RMSD
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
            plt.plot(time_grid, rmsd_values, color=blue, linewidth=2, label='RMSD')
            plt.xlabel('Time (ns)')
            plt.ylabel('RMSD (nm)')
            ax.set_title(f'CA RMSD to Reference - {cfg.method}, {seed}')
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
            # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = analysis_dir / f"rmsd_analysis_{seed}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"RMSD analysis saved to {plot_path}")
            wandb.log({
                "rmsd_analysis": wandb.Image(str(plot_path)),
                "max_rmsd": float(np.max(rmsd_values)),
                "min_rmsd": float(np.min(rmsd_values))
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
    
    try:
        # Load TICA model
        if cfg.molecule == "cln025":
            tica_model_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_model_switch_lag10.pkl"
            cad_full_path = f"./dataset/{cfg.molecule.upper()}-all/cad-switch.pt"
        else:
            tica_model_path = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_tica_model_lag10.pkl"
            cad_full_path = f"./dataset/{cfg.molecule.upper()}-all/cad.pt"
            
        with open(tica_model_path, 'rb') as f:
            tica_model = pickle.load(f)
        
        # Load full dataset for background
        if Path(cad_full_path).exists():
            cad_full = torch.load(cad_full_path)
            tica_coord_full = tica_model.transform(cad_full.numpy())
        else:
            logger.warning(f"Full CAD dataset not found: {cad_full_path}")
            tica_coord_full = None
        
        # Process simulation trajectory
        for seed in range(max_seed + 1):
            # Load trajectory data
            traj_file = log_dir / f"{seed}.xtc"
            if not traj_file.exists()   :
                logger.warning(f"No trajectory file found for seed {seed}")
                continue

            top_file = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_from_mae.pdb"
            if not Path(top_file).exists():
                logger.warning(f"Topology file not found: {top_file}")
                continue
            try:
                traj = md.load(str(traj_file), top=top_file)
                ca_resid_pair = np.array([(a.index, b.index) for a, b in combinations(list(traj.topology.residues), 2)])
                ca_pair_contacts, _ = md.compute_contacts(traj, scheme="ca", contacts=ca_resid_pair, periodic=False)
                ca_pair_contacts_switch = (1 - np.power(ca_pair_contacts / 0.8, 6)) / (1 - np.power(ca_pair_contacts / 0.8, 12))
                tica_coord = tica_model.transform(ca_pair_contacts_switch)
            except Exception as e:
                logger.warning(f"Error processing trajectory for seed {seed}: {e}")
                continue
        
            # Create plot
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            if tica_coord_full is not None:
                h = ax.hist2d(
                    tica_coord_full[:, 0], tica_coord_full[:, 1], 
                    bins=100, norm=LogNorm(), alpha=0.3
                )
                plt.colorbar(h[3], ax=ax, label='Log Density')
            ax.scatter(
                tica_coord[:, 0], tica_coord[:, 1], 
                c=blue, s=2, alpha=0.5,
            )
            ax.set_xlabel("TIC 1")
            ax.set_ylabel("TIC 2")
            ax.set_title(f'TICA Scatter Plot - {cfg.method}')
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = analysis_dir / f"tica_scatter_{seed}.png"
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
    
    
    for seed in range(max_seed + 1):
        try:
            colvar_file = log_dir / f"{seed}" / "COLVAR"
            if not colvar_file.exists():
                logger.warning(f"COLVAR file not found: {colvar_file}")
                continue
            traj_dat = np.genfromtxt(colvar_file, skip_header=1)
            time = traj_dat[:, 0] / 1000
            cv = traj_dat[:, 1]
            
            # Create plots
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.plot(time, cv, label=f"CV", alpha=0.8, linewidth=2, c=blue)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("CV Values")
            ax.set_title(f"CV Evolution - {cfg.method} {seed}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = analysis_dir / f"cv_over_time_{seed}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"CV over time plot saved to {plot_path}")
            wandb.log({
                f"cv_over_time_{seed}": wandb.Image(str(plot_path))
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
    log_dir = base_simulation_dir / cfg.date
    analysis_dir = log_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    config = OmegaConf.to_container(cfg)
    wandb.init(
        project="opes-analysis",
        entity="eddy26",
        tags=cfg.tags if hasattr(cfg, 'tags') else ['analysis'],
        config=config,
    )
    
    try:
        # Post process
        # logger.info("Post processing trajectory...")
        # post_process_trajectory(log_dir, analysis_dir, cfg.seed)
        
        # logger.info("Creating energy files with GROMACS and PLUMED...")
        # compute_energy(log_dir, analysis_dir, cfg.seed)
        
        # # Run analysis functions
        # logger.info("Running CV over time analysis...")
        # plot_cv_over_time(cfg, log_dir, cfg.seed, analysis_dir)
        
        # logger.info("Running RMSD analysis...")
        # plot_rmsd_analysis(cfg, log_dir, cfg.seed, analysis_dir)
        
        # logger.info("Running TICA scatter analysis...")
        # plot_tica_scatter(cfg, log_dir, cfg.seed, analysis_dir)
        
        logger.info("Running free energy analysis...")
        # ref_delta_f = compute_ref_delta_f(cfg)
        # plot_free_energy_curve(cfg, log_dir, cfg.seed, analysis_dir, ref_delta_f=ref_delta_f)
        plot_free_energy_curve(cfg, log_dir, cfg.seed, analysis_dir)
        plot_pmf(cfg, 0.02, log_dir, cfg.seed, analysis_dir)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise
    
    finally:
        wandb.finish()




if __name__ == "__main__":
    main()