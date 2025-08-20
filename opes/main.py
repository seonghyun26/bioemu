"""
OPES Simulation Runner and Analysis Tool

This script runs OPES (On-the-fly Probability Enhanced Sampling) simulations
using GROMACS and PLUMED, then analyzes and logs the results to wandb.
"""

import os
import sys
import hydra
import subprocess
import shutil
import wandb
import logging
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pickle
import torch
from matplotlib.colors import LogNorm
from itertools import combinations

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Add the parent directory to Python path to import analysis functions
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'enhance'))
from src import *
from src.constant import COLORS, FONTSIZE_SMALL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

blue = (70 / 255, 110 / 255, 250 / 255)


class OPESSimulationRunner:
    """Main class for running OPES simulations and analysis"""
    
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        # Config
        self.cfg = cfg
        self.molecule = cfg.molecule
        self.base_simulation_dir = Path("./simulations") / self.molecule / self.cfg.method
        self.step = cfg.step
        self.max_seed = cfg.seed
        self.mlcv_dir = Path("./model") / self.cfg.ckpt_dir
        self.mlcv_path = self.mlcv_dir / f"{self.cfg.ckpt_path}-jit.pt"
        self.datetime = datetime.now().strftime('%m%d_%H%M%S')
        
        # Logging
        os.environ['TZ'] = 'Asia/Seoul'
        self.log_dir = self.base_simulation_dir / self.datetime
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"Setting up simulation in: {self.log_dir}")
    
    def prepare_plumed_config(
        self,
        seed: int,
    ) -> Path:
        seed_dir = self.log_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "fes").mkdir(exist_ok=True)
        
        # Copy base plumed configuration
        source_plumed = Path("./config") / f"{self.cfg.method}.dat"
        target_plumed = seed_dir / "plumed.dat"
        shutil.copy2(source_plumed, target_plumed)
        
        # Copy mlcv models
        source_mlcv_model = self.mlcv_path
        target_mlcv_model = seed_dir / f"{self.cfg.ckpt_path}-jit.pt"
        shutil.copy2(source_mlcv_model, target_mlcv_model)
        
        try:
            with open(target_plumed, 'r') as f:
                content = f.read()
            if self.cfg.method not in ['ref']:
                content = content.replace("FILE=model.pt", f"FILE={self.cfg.ckpt_path}-jit.pt")
            content = content.replace("FILE=", f"FILE={seed_dir}/")
            with open(target_plumed, 'w') as f:
                f.write(content)
            logger.info(f"Modified PLUMED configuration: {target_plumed}")
            
        except Exception as e:
            logger.error(f"Error modifying PLUMED file: {e}")
        
        return target_plumed
    
    def run_gromacs_simulation(
        self,
        seed: int,
        plumed_file: Path
    ) -> bool:
        seed_dir = self.log_dir / str(seed)
        gpu_id = self.cfg.gpu
        
        # GROMACS command
        cmd = [
            "gmx", "mdrun",
            "-s", f"./simulations/{self.molecule}/data/nvt_0.tpr",
            "-deffnm", str(seed_dir),
            "-plumed", str(plumed_file),
            "-nsteps", str(self.step),
            "-ntomp", "1",
            "-nb", "gpu",
            "-bonded", "gpu"
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Running GROMACS simulation for seed {seed} on GPU {gpu_id}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd, 
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"GROMACS simulation started for seed {seed}")
            return process
        except Exception as e:
            logger.error(f"Error starting GROMACS simulation for seed {seed}: {e}")
            return None
    
    def post_process_simulation(self, seed: int) -> bool:
        """Handle post-processing tasks after simulation completes"""
        seed_dir = self.log_dir / str(seed)
        
        # Change permissions
        cmd_permission = ["chmod", "777", "-R", str(seed_dir)]
        
        try:
            process_permission = subprocess.run(
                cmd_permission,
                capture_output=True,
                text=True,
                timeout=30
            )
            if process_permission.returncode == 0:
                logger.info(f"Permissions updated for seed {seed}")
                return True
            else:
                logger.warning(f"Permission change failed for seed {seed}: {process_permission.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Permission change timed out for seed {seed}")
            return False
        except Exception as e:
            logger.error(f"Error changing permissions for seed {seed}: {e}")
            return False
    
    def run_simulations(
        self
    ):
        logger.info(f"Starting {self.max_seed + 1} simulations")
        processes = []
        
        # Start all simulations
        for seed in range(self.max_seed + 1):
            plumed_file = self.prepare_plumed_config(seed)
            process = self.run_gromacs_simulation(seed, plumed_file)
            if process:
                processes.append((seed, process))
            time.sleep(1)  # Small delay between starts
        
        # Wait for all simulations to complete
        logger.info("Waiting for all GROMACS simulations to complete...")
        completed_successfully = []
        failed_simulations = []
        
        for seed, process in processes:
            try:
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    logger.info(f"GROMACS simulation completed successfully for seed {seed}")
                    completed_successfully.append(seed)
                    
                    # Run post-processing for successful simulations
                    self.post_process_simulation(seed)
                else:
                    logger.error(f"GROMACS simulation failed for seed {seed} (return code: {process.returncode})")
                    if stderr:
                        logger.error(f"Error output: {stderr}")
                    failed_simulations.append(seed)
            except Exception as e:
                logger.error(f"Error waiting for simulation {seed}: {e}")
                failed_simulations.append(seed)
        
        logger.info(f"Simulation summary: {len(completed_successfully)} successful, {len(failed_simulations)} failed")
        if completed_successfully:
            logger.info(f"Successful seeds: {completed_successfully}")
        if failed_simulations:
            logger.warning(f"Failed seeds: {failed_simulations}")
        
        logger.info("All GROMACS simulations finished!")
    
    def run_analysis(
        self
    ):
        logger.info("Starting analysis")
        
        # Create analysis directory
        analysis_dir = self.log_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            # self._plot_free_energy_curve(analysis_dir)
            self._plot_rmsd_analysis(analysis_dir)
            self._plot_tica_scatter(analysis_dir)
            self._plot_cv_over_time(analysis_dir)
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
    
    def _plot_free_energy_curve(self, analysis_dir: Path):
        """Plot free energy curve from OPES simulation data"""
        logger.info("Plotting free energy curve...")
        
        try:
            # Collect data from all seeds
            all_delta_fs = []
            
            for seed in range(self.max_seed + 1):
                seed_dir = self.log_dir / str(seed)
                colvar_file = seed_dir / "COLVAR"
                
                if not colvar_file.exists():
                    logger.warning(f"COLVAR file not found for seed {seed}: {colvar_file}")
                    continue
                
                # Load COLVAR data
                try:
                    data = np.loadtxt(colvar_file, comments='#')
                    if len(data) == 0:
                        continue
                        
                    with open(colvar_file, 'r') as f:
                        header = f.readline().strip()
                        keys = header.split()[2:]  # Skip '#' and 'FIELDS'
                    
                    # Get time and phi data
                    time_idx = keys.index('time')
                    phi_idx = keys.index('phi')
                    
                    times = data[:, time_idx]
                    phi_values = data[:, phi_idx]
                    
                    # Calculate delta F over time windows
                    window_size = max(1000, len(phi_values) // 20)  # Adaptive window size
                    delta_fs = []
                    
                    for i in range(window_size, len(phi_values), window_size):
                        window_phi = phi_values[max(0, i-window_size):i]
                        # Simple free energy difference calculation
                        A_count = np.sum(window_phi < 0)
                        B_count = np.sum(window_phi > 0)
                        
                        if A_count > 0 and B_count > 0:
                            # Free energy difference: -kT * ln(N_B/N_A)
                            kT = 2.49  # kJ/mol at 300K
                            delta_f = -kT * np.log(B_count / A_count)
                            delta_fs.append(delta_f)
                        else:
                            delta_fs.append(np.nan)
                    
                    all_delta_fs.append(delta_fs)
                    
                except Exception as e:
                    logger.warning(f"Error processing COLVAR for seed {seed}: {e}")
                    continue
            
            if not all_delta_fs:
                logger.warning("No valid COLVAR data found for free energy analysis")
                return
            
            # Pad arrays to same length and compute statistics
            max_len = max(len(df) for df in all_delta_fs)
            padded_delta_fs = []
            
            for df in all_delta_fs:
                padded = np.full(max_len, np.nan)
                padded[:len(df)] = df
                padded_delta_fs.append(padded)
            
            delta_fs_array = np.array(padded_delta_fs)
            mean_delta_fs = np.nanmean(delta_fs_array, axis=0)
            std_delta_fs = np.nanstd(delta_fs_array, axis=0)
            
            # Create time axis (assuming uniform sampling)
            if len(all_delta_fs) > 0 and len(all_delta_fs[0]) > 0:
                total_time = times[-1] / 1000  # Convert ps to ns
                time_points = np.linspace(0, total_time, len(mean_delta_fs))
                
                # Plot
                plt.figure(figsize=(10, 6))
                mask = ~np.isnan(mean_delta_fs)
                
                if np.any(mask):
                    plt.plot(time_points[mask], mean_delta_fs[mask], 
                            color=COLORS[0], linewidth=2, label='Simulation')
                    plt.fill_between(time_points[mask], 
                                   mean_delta_fs[mask] - std_delta_fs[mask],
                                   mean_delta_fs[mask] + std_delta_fs[mask],
                                   alpha=0.3, color=COLORS[0])
                
                # Reference line (if known)
                ref_delta_f = 10.06  # kJ/mol - known reference for CLN025
                plt.axhline(y=ref_delta_f, color=COLORS[1], linestyle='--', 
                           label='Reference', linewidth=2)
                plt.fill_between(time_points, ref_delta_f - 0.5, ref_delta_f + 0.5,
                               color=COLORS[1], alpha=0.2)
                
                plt.xlabel('Time (ns)')
                plt.ylabel(r'$\Delta F$ (kJ/mol)')
                plt.title(f'Free Energy Difference - {self.cfg.method}')
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
                
        except Exception as e:
            logger.error(f"Error in free energy analysis: {e}")
            raise
    
    def _plot_rmsd_analysis(
        self,
        analysis_dir: Path
    ):
        """Calculate and plot alpha carbon RMSD to reference PDB"""
        logger.info("Calculating alpha carbon RMSD to reference structure...")
        
        try:
            # Load reference structure
            ref_pdb_path = f"/opes/data/{self.molecule.upper()}/folded.pdb"
            ref_traj = md.load_pdb(ref_pdb_path)
            
            # Collect RMSD data from all seeds
            all_rmsds = []
            all_times = []
            
            for seed in range(self.max_seed + 1):
                seed_dir = self.log_dir
                
                # Look for trajectory files (common GROMACS output)
                traj_files = []
                for ext in ['.xtc', '.trr', '.dcd']:
                    traj_file = seed_dir / f"{seed}{ext}"
                    if traj_file.exists():
                        traj_files.append(traj_file)
                
                # Look for coordinate files
                coord_file = seed_dir / f"{seed}.gro"
                if not coord_file.exists():
                    coord_file = seed_dir / f"{seed}.pdb"
                
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
                    colvar_file = seed_dir / "COLVAR"
                    if colvar_file.exists():
                        try:
                            colvar_data = np.loadtxt(colvar_file, comments='#')
                            if len(colvar_data) > 0:
                                with open(colvar_file, 'r') as f:
                                    header = f.readline().strip()
                                    keys = header.split()[2:]
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
            plt.title(f'Alpha Carbon RMSD to Reference - {self.cfg.method}')
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
    
    def _plot_tica_scatter(self, analysis_dir: Path):
        """Plot TICA scatter with simulation trajectory overlay"""
        logger.info("Creating TICA scatter plot...")
        
        try:
            # Load TICA model
            if self.molecule == "cln025":
                tica_model_path = f"/opes/data/{self.molecule.upper()}/{self.molecule.upper()}_tica_model_switch_lag10.pkl"
                cad_full_path = f"/opes/dataset/{self.molecule.upper()}-all/cad-switch.pt"
            else:
                tica_model_path = f"/opes/data/{self.molecule.upper()}/{self.molecule.upper()}_tica_model_lag10.pkl"
                cad_full_path = f"/opes/dataset/{self.molecule.upper()}-all/cad.pt"
                
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
            for seed in range(self.max_seed + 1):
                seed_dir = self.log_dir
                
                # Look for trajectory files
                traj_file = None
                for ext in ['.xtc', '.trr']:
                    potential_file = seed_dir / f"{seed}{ext}"
                    if potential_file.exists():
                        traj_file = potential_file
                        break
                
                if not traj_file:
                    logger.warning(f"No trajectory file found for seed {seed}")
                    continue
                
                # Load topology
                top_file = f"/opes/data/{self.molecule.upper()}/{self.molecule.upper()}_from_mae.pdb"
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
            ax.set_title(f'TICA Scatter Plot - {self.cfg.method}')
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
    
    def _plot_cv_over_time(self, analysis_dir: Path):
        """Plot CV values over time for simulation trajectories"""
        logger.info("Creating CV over time plots...")
        
        try:
            # Load TDA model
            model_path = f"/opes/model/_baseline_/tda-{self.molecule.upper()}-jit.pt"
            if not Path(model_path).exists():
                logger.warning(f"TDA model not found: {model_path}")
                return
            
            model = torch.jit.load(model_path)
            model.eval()
            
            # Process each seed
            all_cv_data = []
            for seed in range(self.max_seed + 1):
                seed_dir = self.log_dir
                
                # Look for trajectory files
                traj_file = None
                for ext in ['.xtc', '.trr']:
                    potential_file = seed_dir / f"{seed}{ext}"
                    if potential_file.exists():
                        traj_file = potential_file
                        break
                
                if not traj_file:
                    continue
                
                # Load topology
                top_file = f"/opes/data/{self.molecule.upper()}/{self.molecule.upper()}_from_mae.pdb"
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
    
    def setup_wandb(
        self
    ):
        config = OmegaConf.to_container(self.cfg)
        config.update({
            'datetime': self.datetime,
            'molecule': self.molecule,
            'simulation_dir': str(self.log_dir)
        })
        wandb.init(
            project="opes",
            entity="eddy26",
            tags=self.cfg.tags,
            config=config,
        )
        
        # Log PLUMED parameters
        params = {}
        plumed_file = Path("./config") / f"{self.cfg.method}.dat"
        inside_enhance_sampling = False
        prefix = ""
        try:
            with open(plumed_file, "r") as f:
                for line in f:
                    line = line.strip()

                    if line.startswith("METAD"):
                        inside_enhance_sampling = True
                        prefix = "metad"
                        continue
                    elif line.startswith("opes"):
                        inside_enhance_sampling = True
                        prefix = "opes"
                        continue

                    if inside_enhance_sampling and (
                        line.startswith("... METAD") or \
                        line.startswith("opes") or \
                        line.startswith("#")
                    ):
                        break

                    if inside_enhance_sampling and line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            params[f"{prefix}/{key.strip()}"] = value.strip()
        except FileNotFoundError:
            logger.warning(f"PLUMED file not found: {plumed_file}")
        
        wandb.config.update(params)
        logger.info("Wandb initialized and PLUMED parameters logged")
    
    def run(
        self
    ):
        try:
            self.setup_wandb()
            self.run_simulations()
            self.run_analysis()
            logger.info("OPES simulation and analysis completed!")
            
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            raise
        
        finally:
            wandb.finish()


@hydra.main(
    config_path="config",
    config_name="basic",
    version_base=None,
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    runner = OPESSimulationRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()