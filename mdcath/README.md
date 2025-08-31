# MDCath Protein Data Preprocessing

This directory contains the preprocessing script for the MDCath protein dataset.

## Overview

The `preprocess.py` script processes protein structure data to:

1. **Extract xyz coordinates**: Loads protein structures and saves the xyz coordinates as PyTorch tensors (`xyz.pt`)
2. **Compute CA distances**: Calculates pairwise distances between all alpha carbon (CA) atoms and saves as PyTorch tensors (`cad.pt`)
3. **TICA analysis**: Performs Time-lagged Independent Component Analysis on CA distances with time lags of 10 and 100, saving the models as pickle files

## Requirements

Install the required dependencies:

```bash
pip install mdtraj torch pyemma numpy tqdm
```

## Data Structure

The script expects the following directory structure:

```
mdcath/data/
├── 1a0aA00/
│   ├── 1a0aA00.pdb  (or other structure files)
│   └── ...
├── 1a02F00/
│   ├── 1a02F00.pdb
│   └── ...
└── ...
```

## Supported File Formats

The script automatically detects and loads various protein structure formats:
- PDB files (`.pdb`)
- XTC trajectory files (`.xtc`)
- DCD trajectory files (`.dcd`)
- HDF5 files (`.h5`)
- NetCDF files (`.netcdf`)

## Usage

### Process All Proteins

```bash
python preprocess.py
```

### Test Single Protein

```python
from preprocess import test_single_protein
test_single_protein("1a0aA00")
```

## Output Files

For each protein (e.g., `1a0aA00`), the script generates:

- `xyz.pt`: PyTorch tensor containing xyz coordinates (shape: n_frames × n_atoms × 3)
- `cad.pt`: PyTorch tensor containing CA pairwise distances (shape: n_frames × n_pairs)
- `tica_model_lag_10.pkl`: TICA model with time lag 10
- `tica_model_lag_100.pkl`: TICA model with time lag 100

## TICA Analysis

The TICA (Time-lagged Independent Component Analysis) follows the pattern from the CLN025 notebook:

```python
tica_obj = pyemma.coordinates.tica(ca_distances, lag=lag, dim=2)
```

- **Input**: CA pairwise distances
- **Time lags**: 10 and 100 frames
- **Dimensions**: 2 (as used in the reference notebook)

## Error Handling

The script includes comprehensive error handling:
- Checks for missing dependencies
- Validates file existence and format
- Ensures sufficient frames for TICA analysis
- Provides detailed error messages and warnings

## Example Usage in Python

```python
import torch
import pickle

# Load processed data
protein_id = "1a0aA00"
data_dir = "mdcath/data"

# Load xyz coordinates
xyz = torch.load(f"{data_dir}/{protein_id}/xyz.pt")
print(f"XYZ shape: {xyz.shape}")

# Load CA distances
ca_distances = torch.load(f"{data_dir}/{protein_id}/cad.pt")
print(f"CA distances shape: {ca_distances.shape}")

# Load TICA model
with open(f"{data_dir}/{protein_id}/tica_model_lag_10.pkl", 'rb') as f:
    tica_model = pickle.load(f)
    
# Get TICA projection
tica_data = tica_model.get_output()[0]
print(f"TICA data shape: {tica_data.shape}")
```