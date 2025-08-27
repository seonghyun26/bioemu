import os
import sys
import torch
import hydra
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pickle
import wandb
import logging
from matplotlib.colors import LogNorm
from itertools import combinations
from pathlib import Path
from omegaconf import OmegaConf

# Add the parent directory to Python path to import analysis functions
from src import *
from src.constant import COLORS, FONTSIZE_SMALL

# Set up logging
logger = logging.getLogger(__name__)

blue = (70 / 255, 110 / 255, 250 / 255)


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


def plot_free_energy_curve(cfg, log_dir: Path, max_seed: int, analysis_dir: Path):
    """Plot free energy curve from OPES simulation data using folded/unfolded state analysis"""
    logger.info("Plotting free energy curve using hydrogen bond-based folded/unfolded states...")
    
    # try:
    # Collect data from all seeds
    all_delta_fs = []
    all_times = []
    
    for seed in range(max_seed + 1):
        # Look for COLVAR file directly in log_dir
        colvar_file = log_dir / f"{seed}" / "COLVAR"
        
        if not colvar_file.exists():
            logger.warning(f"COLVAR file not found: {colvar_file}")
            continue
        
        # Load COLVAR data to get time information
        try:
            colvar_data = np.loadtxt(colvar_file, comments='#')
            if len(colvar_data) == 0:
                continue
                
            with open(colvar_file, 'r') as f:
                header = f.readline().strip()
                keys = header.split()[2:]  # Skip '#!' and 'FIELDS'
            
            # Get time data
            time_idx = keys.index('time')
            times = colvar_data[:, time_idx] / 1000  # Convert ps to ns
            
            # Load trajectory for hydrogen bond analysis
            traj_file = None
            for ext in ['.xtc', '.trr', '.dcd']:
                potential_file = log_dir / f"{seed}{ext}"
                if potential_file.exists():
                    traj_file = potential_file
                    break
            
            if not traj_file:
                logger.warning(f"No trajectory file found for seed {seed}")
                continue
            
            # Load topology
            top_file = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_from_mae.pdb"
            if not Path(top_file).exists():
                logger.warning(f"Topology file not found: {top_file}")
                continue
            
            # Load trajectory
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
    plt.figure(figsize=(10, 6))
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


def plot_rmsd_analysis(cfg, log_dir: Path, max_seed: int, analysis_dir: Path):
    """Calculate and plot alpha carbon RMSD to reference PDB"""
    logger.info("Calculating alpha carbon RMSD to reference structure...")
    
    try:
        # Load reference structure
        ref_pdb_path = f"./data/{cfg.molecule.upper()}/folded.pdb"
        ref_traj = md.load_pdb(ref_pdb_path)
        
        # Collect RMSD data from all seeds
        all_rmsds = []
        all_times = []
        
        for seed in range(max_seed + 1):
            # Look for trajectory files (common GROMACS output)
            traj_files = []
            for ext in ['.xtc', '.trr', '.dcd']:
                traj_file = log_dir / f"{seed}{ext}"
                if traj_file.exists():
                    traj_files.append(traj_file)
            
            # Look for coordinate files
            coord_file = log_dir / f"{seed}.gro"
            if not coord_file.exists():
                coord_file = log_dir / f"{seed}.pdb"
            
            if not traj_files and not coord_file.exists():
                logger.warning(f"No trajectory files found for seed {seed}")
                continue
            
            try:
                # Load trajectory - try different formats
                traj = None
                if traj_files:
                    # Use first available trajectory file
                    traj_file = traj_files[0]
                    traj = md.load(str(traj_file), top=str(ref_pdb_path))
                elif coord_file.exists():
                    # Single frame from coordinate file
                    traj = md.load(str(coord_file))
                
                if traj is None:
                    continue
                
                # Compute RMSD
                rmsd_values = md.rmsd(
                    traj,
                    ref_traj,
                    atom_indices = traj.topology.select("name CA")                        
                )
                all_rmsds.append(rmsd_values)
                
                # Create time array (assuming uniform time steps)
                # Get time info from COLVAR if available
                colvar_file = log_dir / "COLVAR"
                if colvar_file.exists():
                    try:
                        colvar_data = np.loadtxt(colvar_file, comments='#')
                        if len(colvar_data) > 0:
                            with open(colvar_file, 'r') as f:
                                header = f.readline().strip()
                                keys = header.split()[2:]  # Skip '#!' and 'FIELDS'
                            time_idx = keys.index('time')
                            times = colvar_data[:, time_idx] / 1000  # Convert ps to ns
                            # Interpolate to match trajectory length
                            if len(times) != len(rmsd_values):
                                times = np.linspace(0, times[-1], len(rmsd_values))
                            all_times.append(times)
                        else:
                            # Fallback: assume 1 ps timestep
                            times = np.arange(len(rmsd_values)) * 0.001  # 1 ps = 0.001 ns
                            all_times.append(times)
                    except:
                        # Fallback: assume 1 ps timestep
                        times = np.arange(len(rmsd_values)) * 0.001
                        all_times.append(times)
                else:
                    # Fallback: assume 1 ps timestep
                    times = np.arange(len(rmsd_values)) * 0.001
                    all_times.append(times)
                
            except Exception as e:
                logger.warning(f"Error processing trajectory for seed {seed}: {e}")
                continue
        
        if not all_rmsds:
            logger.warning("No valid trajectory data found for RMSD analysis")
            return
        
        # Find common time range
        min_length = min(len(rmsd) for rmsd in all_rmsds)
        if min_length == 0:
            logger.warning("All trajectories are empty")
            return
        
        # Truncate all arrays to same length
        truncated_rmsds = [rmsd[:min_length] for rmsd in all_rmsds]
        truncated_times = [times[:min_length] for times in all_times]
        
        # Convert to arrays and compute statistics
        rmsd_array = np.array(truncated_rmsds)
        mean_rmsd = np.mean(rmsd_array, axis=0)
        std_rmsd = np.std(rmsd_array, axis=0)
        
        # Use time from first seed (should be similar for all)
        time_axis = truncated_times[0] if truncated_times else np.arange(min_length) * 0.001
        
        # Plot RMSD
        plt.figure(figsize=(5, 3))
        
        plt.plot(time_axis, mean_rmsd, color=blue, linewidth=2, label='Mean RMSD')
        plt.fill_between(time_axis, mean_rmsd - std_rmsd, mean_rmsd + std_rmsd,
                       alpha=0.3, color=COLORS[2])
        
        # Plot individual trajectories with transparency
        for i, rmsd in enumerate(truncated_rmsds):
            plt.plot(truncated_times[i] if i < len(truncated_times) else time_axis, 
                    rmsd, alpha=0.3, color='gray', linewidth=2)
        
        from matplotlib.ticker import FormatStrFormatter
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        plt.xlabel('Time (ns)')
        plt.ylabel('RMSD (nm)')
        plt.title(f'Alpha Carbon RMSD to Reference - {cfg.method}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = analysis_dir / "rmsd_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"RMSD analysis saved to {plot_path}")
        
        # Log to wandb
        wandb.log({
            "rmsd_analysis": wandb.Image(str(plot_path)),
            "final_rmsd_mean": float(mean_rmsd[-1]),
            "final_rmsd_std": float(std_rmsd[-1]),
            "max_rmsd": float(np.max(mean_rmsd)),
            "min_rmsd": float(np.min(mean_rmsd))
        })
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in RMSD analysis: {e}")
        raise


def plot_tica_scatter(cfg, log_dir: Path, max_seed: int, analysis_dir: Path):
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
        simulation_tica_coords = []
        for seed in range(max_seed + 1):
            # Look for trajectory files
            traj_file = None
            for ext in ['.xtc', '.trr']:
                potential_file = log_dir / f"{seed}{ext}"
                if potential_file.exists():
                    traj_file = potential_file
                    break
            
            if not traj_file:
                logger.warning(f"No trajectory file found for seed {seed}")
                continue
            
            # Load topology
            top_file = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_from_mae.pdb"
            if not Path(top_file).exists():
                logger.warning(f"Topology file not found: {top_file}")
                continue
            
            try:
                # Load trajectory
                traj = md.load(str(traj_file), top=top_file)
                
                # Compute contacts
                ca_resid_pair = np.array([(a.index, b.index) for a, b in combinations(list(traj.topology.residues), 2)])
                ca_pair_contacts, _ = md.compute_contacts(traj, scheme="ca", contacts=ca_resid_pair, periodic=False)
                
                # Apply switch function
                ca_pair_contacts_switch = (1 - np.power(ca_pair_contacts / 0.8, 6)) / (1 - np.power(ca_pair_contacts / 0.8, 12))
                
                # Transform to TICA coordinates
                tica_coord = tica_model.transform(ca_pair_contacts_switch)
                simulation_tica_coords.append(tica_coord)
                
            except Exception as e:
                logger.warning(f"Error processing trajectory for seed {seed}: {e}")
                continue
        
        if not simulation_tica_coords:
            logger.warning("No valid simulation trajectories found for TICA analysis")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot background (full dataset) if available
        if tica_coord_full is not None:
            h = ax.hist2d(
                tica_coord_full[:, 0], tica_coord_full[:, 1], 
                bins=100, norm=LogNorm(), alpha=0.3
            )
            plt.colorbar(h[3], ax=ax, label='Log Density')
        
        # Plot simulation trajectories
        colors = plt.cm.Set1(np.linspace(0, 1, len(simulation_tica_coords)))
        for i, tica_coord in enumerate(simulation_tica_coords):
            ax.scatter(
                tica_coord[:, 0], tica_coord[:, 1], 
                c=blue, s=4, alpha=0.5, label=f'Seed {i}',
            )
        
        ax.set_xlabel("TIC 1")
        ax.set_ylabel("TIC 2")
        ax.set_title(f'TICA Scatter Plot - {cfg.method}')
        if len(simulation_tica_coords) <= 5:  # Only show legend if not too many seeds
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = analysis_dir / "tica_scatter.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"TICA scatter plot saved to {plot_path}")
        
        # Log to wandb
        wandb.log({"tica_scatter": wandb.Image(str(plot_path))})
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in TICA scatter analysis: {e}")
        raise


def plot_cv_over_time(cfg, log_dir: Path, max_seed: int, analysis_dir: Path, mlcv_path: Path):
    """Plot CV values over time for simulation trajectories"""
    logger.info("Creating CV over time plots...")
    
    try:
        # Load TDA model
        model_path = f"{mlcv_path}/{cfg.ckpt_path}-jit.pt"
        if not Path(model_path).exists():
            logger.warning(f"{cfg.method} model not found: {model_path}")
            return
        
        model = torch.jit.load(model_path)
        model.eval()
        
        # Process each seed
        all_cv_data = []
        for seed in range(max_seed + 1):
            # Look for trajectory files
            traj_file = None
            for ext in ['.xtc', '.trr']:
                potential_file = log_dir / f"{seed}{ext}"
                if potential_file.exists():
                    traj_file = potential_file
                    break
            
            if not traj_file:
                continue
            
            # Load topology
            top_file = f"./data/{cfg.molecule.upper()}/{cfg.molecule.upper()}_from_mae.pdb"
            if not Path(top_file).exists():
                continue
            
            try:
                # Load trajectory
                traj = md.load(str(traj_file), top=top_file)
                
                # Compute contacts
                ca_resid_pair = np.array([(a.index, b.index) for a, b in combinations(list(traj.topology.residues), 2)])
                ca_pair_contacts, _ = md.compute_contacts(traj, scheme="ca", contacts=ca_resid_pair, periodic=False)
                
                # Compute CV values
                with torch.no_grad():
                    cv = model(torch.from_numpy(ca_pair_contacts))
                    cv_np = cv.detach().numpy()
                
                all_cv_data.append((seed, cv_np))
                
            except Exception as e:
                logger.warning(f"Error processing CV for seed {seed}: {e}")
                continue
        
        if not all_cv_data:
            logger.warning("No valid CV data found")
            return
        
        # Create plots
        n_seeds = len(all_cv_data)
        fig, axes = plt.subplots(n_seeds, 1, figsize=(5, 3), squeeze=False)
        
        for i, (seed, cv_np) in enumerate(all_cv_data):
            ax = axes[i, 0]
            n_frames, n_cvs = cv_np.shape
            time = np.arange(n_frames)
            
            for j in range(n_cvs):
                ax.plot(time, cv_np[:, j], label=f"CV {j}", alpha=0.8, linewidth=2, c=blue)
            
            ax.set_xlabel("Frames")
            ax.set_ylabel("CV Values")
            ax.set_title(f"CV Evolution - Seed {seed}")
            if n_cvs > 1:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = analysis_dir / "cv_over_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"CV over time plot saved to {plot_path}")
        
        # Log to wandb
        wandb.log({"cv_over_time": wandb.Image(str(plot_path))})
        
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
    base_simulation_dir = Path("./simulations") / cfg.molecule / cfg.method
    log_dir = base_simulation_dir / cfg.date
    analysis_dir = log_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    mlcv_dir = Path("./model") / cfg.ckpt_dir
    
    # Initialize wandb
    config = OmegaConf.to_container(cfg)
    wandb.init(
        project="opes-analysis",
        entity="eddy26",
        tags=cfg.tags if hasattr(cfg, 'tags') else ['analysis'],
        config=config,
    )
    
    try:
        # Run analysis functions
        logger.info("Running free energy analysis...")
        plot_free_energy_curve(cfg, log_dir, cfg.seed, analysis_dir)
        
        logger.info("Running RMSD analysis...")
        plot_rmsd_analysis(cfg, log_dir, cfg.seed, analysis_dir)
        
        logger.info("Running TICA scatter analysis...")
        plot_tica_scatter(cfg, log_dir, cfg.seed, analysis_dir)
        
        logger.info("Running CV over time analysis...")
        plot_cv_over_time(cfg, log_dir, cfg.seed, analysis_dir, mlcv_dir)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise
    
    finally:
        wandb.finish()




if __name__ == "__main__":
    main()