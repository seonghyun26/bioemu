#!/usr/bin/env python3
"""
Test script to demonstrate parallel execution concept used in main.py

This script shows how multiple processes can be started in parallel and then
waited for completion, which is exactly what the main.py does with GROMACS simulations.
"""

import subprocess
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_dummy_process(seed: int, duration: int = 5) -> subprocess.Popen:
    """Run a dummy process that sleeps for the specified duration"""
    cmd = ["sleep", str(duration)]
    logger.info(f"Starting dummy process for seed {seed} (will run for {duration}s)")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Dummy process started for seed {seed} (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Error starting dummy process for seed {seed}: {e}")
        return None

def run_parallel_simulations(num_seeds: int = 3):
    """Demonstrate parallel execution similar to main.py"""
    logger.info(f"Starting {num_seeds} dummy simulations in parallel")
    processes = []
    
    # Start all simulations (this is non-blocking)
    for seed in range(num_seeds):
        process = run_dummy_process(seed, duration=5 + seed)  # Different durations for demo
        if process:
            processes.append((seed, process))
        time.sleep(0.5)  # Small delay between starts
    
    if not processes:
        logger.error("No processes were started successfully")
        return
    
    logger.info(f"Successfully started {len(processes)} processes")
    
    # Wait for all processes to complete (this is where we wait)
    logger.info("Waiting for all processes to complete...")
    completed_successfully = []
    failed_processes = []
    
    total_processes = len(processes)
    for i, (seed, process) in enumerate(processes):
        try:
            logger.info(f"Waiting for seed {seed} to complete... ({i+1}/{total_processes})")
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                logger.info(f"Process completed successfully for seed {seed}")
                completed_successfully.append(seed)
            else:
                logger.error(f"Process failed for seed {seed} (return code: {process.returncode})")
                failed_processes.append(seed)
        except Exception as e:
            logger.error(f"Error waiting for process {seed}: {e}")
            failed_processes.append(seed)
    
    logger.info(f"Process summary: {len(completed_successfully)} successful, {len(failed_processes)} failed")
    if completed_successfully:
        logger.info(f"Successful seeds: {completed_successfully}")
    if failed_processes:
        logger.warning(f"Failed seeds: {failed_processes}")
    
    logger.info("All processes finished!")

if __name__ == "__main__":
    logger.info("=== Testing Parallel Execution Concept ===")
    logger.info("This demonstrates how main.py runs multiple GROMACS simulations in parallel")
    logger.info("Key points:")
    logger.info("1. All processes are started first (non-blocking)")
    logger.info("2. Then we wait for each one to complete")
    logger.info("3. This allows true parallel execution")
    logger.info("")
    
    run_parallel_simulations(3)
