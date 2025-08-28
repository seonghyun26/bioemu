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
import re
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
from analysis_opes import (
    plot_free_energy_curve,
    plot_rmsd_analysis,
    plot_tica_scatter,
    plot_cv_over_time
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        self.datetime = cfg.date
        self.background = cfg.get("background", False)
        
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
        
        # Create symbolic link for base plumed configuration
        source_plumed = Path("./config") / f"{self.cfg.method}.dat"
        target_plumed = seed_dir / "plumed.dat"
        shutil.copy2(source_plumed, target_plumed)
        # if target_plumed.exists() or target_plumed.is_symlink():
        #     target_plumed.unlink()
        # target_plumed.symlink_to(source_plumed.resolve())
        
        # Create symbolic link for mlcv models
        source_mlcv_model = self.mlcv_path
        target_mlcv_model = seed_dir / f"{self.cfg.ckpt_path}-jit.pt"
        shutil.copy2(source_mlcv_model, target_mlcv_model)
        # if target_mlcv_model.exists() or target_mlcv_model.is_symlink():
        #     target_mlcv_model.unlink()
        # target_mlcv_model.symlink_to(source_mlcv_model.resolve())
        
        try:
            with open(target_plumed, 'r') as f:
                content = f.read()
            if self.cfg.method not in ['ref']:
                content = content.replace("FILE=model.pt", f"FILE={self.cfg.ckpt_path}-jit.pt")
            content = content.replace("FILE=", f"FILE={seed_dir}/")
            
            # Replace SIGMA value if specified in config
            if hasattr(self.cfg, 'sigma'):
                # Use regex to find and replace SIGMA parameter more robustly
                sigma_pattern = r'SIGMA=[\d\.]+' 
                content = re.sub(sigma_pattern, f'SIGMA={self.cfg.sigma}', content)
                logger.info(f"Set SIGMA parameter to {self.cfg.sigma}")
            wandb.config.update(
                {
                    'opes/SIGMA': str(self.cfg.sigma)
                },
                allow_val_change=True
            )
            
            with open(target_plumed, 'w') as f:
                f.write(content)
            logger.info(f"Created symbolic links and modified PLUMED configuration: {target_plumed}")
            
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
            "-reseed", str(seed),
            "-ntomp", "1",
            "-nb", "gpu",
            "-bonded", "gpu",
        ]
        if self.background:
            cmd += [ "&" ]
        
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
            # plot_free_energy_curve(self.cfg, self.log_dir, self.max_seed, analysis_dir)
            plot_rmsd_analysis(self.cfg, self.log_dir, self.max_seed, analysis_dir)
            plot_tica_scatter(self.cfg, self.log_dir, self.max_seed, analysis_dir)
            plot_cv_over_time(self.cfg, self.log_dir, self.max_seed, analysis_dir, self.mlcv_dir)
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
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
        
        wandb.config.update(params, allow_val_change=True)
        logger.info("Wandb initialized and PLUMED parameters logged")
    
    def run(
        self
    ):
        try:
            self.setup_wandb()
            self.run_simulations()
            # self.run_analysis()
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