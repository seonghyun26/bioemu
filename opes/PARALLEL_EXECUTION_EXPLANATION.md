# Parallel Execution in OPES Main.py

## Overview

The `main.py` script is designed to run multiple GROMACS simulations in parallel, one for each seed value from 0 to `max_seed`. This document explains how the parallel execution works and why it's effective.

## How It Works

### 1. Sequential Startup, Parallel Execution

```python
# Start all simulations
for seed in range(self.max_seed + 1):
    plumed_file = self.prepare_plumed_config(seed)
    process = self.run_gromacs_simulation(seed, plumed_file)
    if process:
        processes.append((seed, process))
    time.sleep(1)  # Small delay between starts
```

**Key Point**: The simulations are started one by one in a loop, but each call to `run_gromacs_simulation()` returns immediately after starting the process. The `subprocess.Popen()` call is **non-blocking**.

### 2. Process Management

```python
def run_gromacs_simulation(self, seed: int, plumed_file: Path):
    # ... command preparation ...
    process = subprocess.Popen(
        cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subIPE,
        text=True
    )
    return process  # Returns immediately, doesn't wait for completion
```

**Key Point**: `subprocess.Popen()` starts the process and returns a process object immediately. The GROMACS simulation continues running in the background.

### 3. Parallel Waiting

```python
# Wait for all simulations to complete
for seed, process in processes:
    stdout, stderr = process.communicate()  # This waits for THIS process
    # ... handle completion ...
```

**Key Point**: `process.communicate()` waits for each individual process to complete, but since all processes are already running, they complete independently.

## Why This Approach Works

### 1. True Parallelism
- All GROMACS processes run simultaneously
- CPU and GPU resources are shared among processes
- Total execution time ≈ time of longest simulation (not sum of all)

### 2. Resource Sharing
- Multiple GROMACS processes can share the same GPU
- GROMACS handles GPU memory management internally
- Each process gets its own working directory and files

### 3. Process Control
- Each process can be monitored individually
- Failed processes don't affect others
- Easy to implement retry logic if needed

## Example Timeline

```
Time 0s:   Start seed 0 simulation
Time 1s:   Start seed 1 simulation  
Time 2s:   Start seed 2 simulation
Time 3s:   All simulations running in parallel
Time 4s:   All simulations running in parallel
...
Time 60s:  Seed 0 completes
Time 65s:  Seed 1 completes  
Time 70s:  Seed 2 completes
Time 70s:  All done!
```

## GPU Handling

All simulations use the same GPU ID (`self.cfg.gpu`). This works because:

1. **GROMACS GPU Sharing**: GROMACS can handle multiple processes on the same GPU
2. **Memory Management**: Each simulation gets its own GPU memory allocation
3. **Scheduling**: The GPU driver handles process scheduling

## Monitoring and Logging

The script provides comprehensive monitoring:

- **Process IDs**: Each simulation gets a unique PID for tracking
- **Progress Tracking**: Shows which simulation is being waited for
- **Success/Failure Tracking**: Maintains lists of successful and failed simulations
- **Error Handling**: Captures stdout/stderr for debugging

## Potential Improvements

1. **GPU Load Balancing**: Could assign different GPUs to different simulations
2. **Resource Monitoring**: Could add CPU/GPU usage monitoring
3. **Dynamic Scaling**: Could start simulations based on available resources
4. **Fault Tolerance**: Could add automatic retry for failed simulations

## Conclusion

The current implementation correctly implements parallel execution of multiple GROMACS simulations. The key insight is that `subprocess.Popen()` starts processes without waiting, allowing true parallel execution while maintaining full control and monitoring capabilities.
