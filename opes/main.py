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

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Add the parent directory to Python path to import analysis functions
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'enhance'))
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
        self.step = cfg.step
        self.max_seed = cfg.seed
        self.mlcv_path = Path("./") / f"{self.cfg.ckpt_path}-jit.pt"
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
        target_mlcv_model = seed_dir / self.mlcv_path
        shutil.copy2(source_mlcv_model, target_mlcv_model)
        
        try:
            with open(target_plumed, 'r') as f:
                content = f.read()
            if self.cfg.method not in ['ref']:
                content = content.replace("FILE=model.pt", f"FILE={self.mlcv_path}")
                content = content.replace("FILE=", f"FILE={seed_dir}/")
            else:
                content = content.replace("FILE=", f"FILE={seed_dir}/")            
            with open(target_plumed, 'w') as f:
                f.write(content)
            logger.info(f"Modified PLUMED configuration: {target_plumed}")
            
        except Exception as e:
            logger.error(f"Error modifying PLUMED file: {e}")
            
        return target_plumed
    
    def run_gromacs_simulation(self, seed: int, plumed_file: Path) -> bool:
        """Run GROMACS simulation with PLUMED for a specific seed"""
        seed_dir = self.log_dir / str(seed)
        gpu_id = seed
        
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
            return process
        except Exception as e:
            logger.error(f"Error starting GROMACS simulation for seed {seed}: {e}")
            return None
    
    def run_simulations(self):
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
        for seed, process in processes:
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                logger.info(f"GROMACS simulation completed successfully for seed {seed}")
            else:
                logger.error(f"GROMACS simulation failed for seed {seed}")
                logger.error(f"Error output: {stderr}")
        
        logger.info("All GROMACS simulations finished!")
    
    def run_analysis(self):
        logger.info("Starting analysis and plotting...")
        base_dir = str(self.log_dir)
        
        # # Plot free energy difference
        # logger.info("Plotting free energy difference") 
        # plot_free_energy_difference(self.args, base_dir)
        
        # # Plot FES over CV (if not reference method)
        # if self.args.method != "ref":
        #     logger.info("Plotting FES over CV")
        #     plot_fes_over_cv(self.args, base_dir)
    
    def setup_wandb(self):
        config = OmegaConf.to_container(self.cfg)
        config.update({
            'datetime': self.datetime,
            'molecule': self.molecule,
            'simulation_dir': str(self.log_dir)
        })
        plumed_file = self.base_simulation_dir / "plumed.dat"
        
        wandb.init(
            project="opes",
            entity="eddy26",
            tags=list(self.cfg.tags),
            config=config,
        )
        
        # Log PLUMED parameters
        params = {}
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

                    if inside_enhance_sampling and line.startswith("... METAD"):
                        break
                    elif inside_enhance_sampling and line.startswith("opes"):
                        break

                    if inside_enhance_sampling and line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            params[f"{prefix}/{key.strip()}"] = value.strip()
        except FileNotFoundError:
            logger.warning(f"PLUMED file not found: {plumed_file}")
        
        wandb.log(params)
        logger.info("Wandb initialized and PLUMED parameters logged")
    
    def run(self):
        try:
            self.setup_wandb()
            self.run_simulations()
            self.run_analysis()
            logger.info("OPES simulation and analysis completed successfully!")
            
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