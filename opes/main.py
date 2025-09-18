"""
OPES Simulation Runner and Analysis Tool

This script runs OPES (On-the-fly Probability Enhanced Sampling) simulations
using GROMACS and PLUMED, then analyzes and logs the results to wandb.

PARALLEL EXECUTION APPROACH:
- The script starts multiple GROMACS simulations in parallel (one for each seed)
- Each simulation runs independently using subprocess.Popen (non-blocking)
- After starting all simulations, the script waits for each one to complete
- This approach allows true parallel execution while maintaining control and monitoring
- All simulations share the same GPU (GROMACS handles GPU sharing internally)
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

from src import *

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
        self.step = cfg.opes.step
        self.max_seed = cfg.opes.max_seed
        self.sigma = cfg.opes.sigma
        self.mlcv_dir = Path("./model") / self.cfg.ckpt_dir
        self.mlcv_path = self.mlcv_dir / f"{self.cfg.ckpt_path}-jit.pt"
        self.datetime = cfg.date
        
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
        source_plumed = Path("./config") / f"{self.cfg.method}-{self.molecule}.dat"
        target_plumed = seed_dir / "plumed.dat"
        shutil.copy2(source_plumed, target_plumed)
        logger.info(f"Copied PLUMED configuration: {source_plumed} to {target_plumed}")
        # if target_plumed.exists() or target_plumed.is_symlink():
        #     target_plumed.unlink()
        # target_plumed.symlink_to(source_plumed.resolve())
        
        # Create symbolic link for mlcv models
        source_mlcv_model = self.mlcv_path
        target_mlcv_model = seed_dir / f"{self.cfg.ckpt_path}-jit.pt"
        shutil.copy2(source_mlcv_model, target_mlcv_model)
        jit_model = torch.jit.load(target_mlcv_model)
        jit_model.eval()
        for name, param in jit_model.named_parameters():
            if param.device != torch.device(f"cuda:{seed}"):
                jit_model = jit_model.to(f"cuda:{seed}")
                torch.jit.save(jit_model, target_mlcv_model)
                logger.info(f"{target_mlcv_model} moved to cuda:{seed}")
            else:
                logger.info(f"{target_mlcv_model} already at cuda")
            break
        logger.info(f"Copied MLCV model: {source_mlcv_model} to {target_mlcv_model}")
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
                content = re.sub(sigma_pattern, f'SIGMA={self.sigma}', content)
                logger.info(f"Set SIGMA parameter to {self.sigma}")
            wandb.config.update(
                {
                    'opes/SIGMA': str(self.sigma)
                },
                allow_val_change=True
            )
            
            with open(target_plumed, 'w') as f:
                f.write(content)
            logger.info(f"Copied model and modified PLUMED configuration: {target_plumed}")
            
        except Exception as e:
            logger.error(f"Error modifying PLUMED file: {e}")
        
        return target_plumed
    
    def run_gromacs_simulation(
        self,
        seed: int,
        plumed_file: Path
    ) -> bool:
        seed_dir = self.log_dir / str(seed)
        gpu_id = seed + self.cfg.start_gpu
        
        # GROMACS command
        # if self.cfg.gpu == 0:
        #     logger.info(f"Running GROMACS simulation for seed {seed} on CPU")
        #     cmd = [
        #         "gmx", "mdrun",
        #         "-s", f"./data/{self.molecule.upper()}/nvt_0.tpr",
        #         "-deffnm", str(seed_dir),
        #         "-plumed", str(plumed_file),
        #         "-nsteps", str(self.step),
        #         "-reseed", str(seed),
        #         "-ntomp", "1",
        #         "-bonded", "cpu",
        #         "-nb", "cpu",
        #         "-pin", "on",
        #         "-pme", "cpu",
        #     ]
        # else:
            # logger.info(f"Running GROMACS simulation for seed {seed} on GPU")
        # if self.cfg.molecule == "1fme":
        #     nvt_name = "nvt_ions"
        # elif self.cfg.molecule == "gtt":
        #     nvt_name = "nvt_1"
        # else:
        #     nvt_name = "nvt_0"
        if self.cfg.molecule in ["cln025", "2jof", "1fme", "gtt"]:
            nvt_name = "md"
        else:
            nvt_name = "nvt_0"
        ntomp_num = self.cfg.opes.ntomp
        cmd = [
            "gmx", "mdrun",
            "-s", f"./data/{self.molecule.upper()}/{nvt_name}.tpr",
            "-deffnm", str(seed_dir),
            "-plumed", str(plumed_file),
            "-nsteps", str(self.step),
            "-reseed", str(seed),
            "-ntomp", str(ntomp_num),
            "-bonded", "gpu",
            "-nb", "gpu",
            "-pme", "gpu",
            "-pin", "on",
            "-pinoffset", str(gpu_id * ntomp_num),
            "-tunepme",
            # "-pinstride", "1",
            # "-dlb", "no"
            # "-update", "gpu",
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['OMP_NUM_THREADS'] = str(ntomp_num)
        env['GMX_CUDA_GRAPH'] = "1"
        seed_dir = Path(self.log_dir) / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = open(seed_dir / "mdrun.out", "wb", buffering=0)
        stderr_file = open(seed_dir / "mdrun.err", "wb", buffering=0)
        
        logger.info(f"\nRunning GROMACS simulation for seed {seed} on GPU {gpu_id}")
        logger.info(f"Command: {' '.join(cmd)}")
        try:
            process = subprocess.Popen(
                cmd, 
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=False,
                close_fds=True,
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
        """
        Run multiple GROMACS simulations in parallel for different seeds.
        Each simulation runs independently and all are monitored until completion.
        """
        logger.info(f"Starting {self.max_seed + 1} simulations in parallel")
        processes = []
        
        # Start all simulations
        for seed in range(self.max_seed + 1):
            plumed_file = self.prepare_plumed_config(seed)
            process = self.run_gromacs_simulation(seed, plumed_file)
            if process:
                processes.append((seed, process))
                logger.info(f"Started simulation for seed {seed} (PID: {process.pid})")
            else:
                logger.error(f"Failed to start simulation for seed {seed}")
            time.sleep(1)  # Small delay between starts
        
        if not processes:
            logger.error("No simulations were started successfully")
            return
        
        logger.info(f"Successfully started {len(processes)} simulations")
        
        # Wait for all simulations to complete
        logger.info("Waiting for all GROMACS simulations to complete...")
        completed_successfully = []
        failed_simulations = []
        
        # Monitor all processes
        total_processes = len(processes)
        for i, (seed, process) in enumerate(processes):
            try:
                logger.info(f"Waiting for seed {seed} to complete... ({i+1}/{total_processes})")
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
        plumed_file = Path("./config") / f"{self.cfg.method}-{self.molecule}.dat"
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
            logger.info("OPES simulation completed!")
            
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