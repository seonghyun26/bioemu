"""
Preprocessing script for protein data in mdcath dataset.

This script processes protein structures to:
1. Extract xyz coordinates and save as PyTorch tensors
2. Compute CA-wise distances and save as PyTorch tensors  
3. Perform TICA analysis on CA distances with time lags 10 and 100
4. Save TICA models using pickle

Requirements:
- mdtraj: pip install mdtraj
- torch: pip install torch
- pyemma: pip install pyemma
- numpy: pip install numpy
- tqdm: pip install tqdm

Usage:
    python preprocess.py
"""

import mdtraj as md
import numpy as np
import torch
import pickle
import pyemma
import os
from pathlib import Path
from tqdm import tqdm


def load_protein_data(protein_id, data_dir="/home/shpark/prj-mlcv/lib/mdcath/data"):
    """
    Load protein structure data for a given protein ID.
    Supports multiple file formats: PDB, XTC, DCD, etc.
    """
    protein_dir = Path(data_dir) / protein_id
    
    if not protein_dir.exists():
        raise FileNotFoundError(f"Protein directory {protein_dir} does not exist")
    
    # Look for common protein file formats in order of preference
    possible_files = [
        # PDB files
        protein_dir / f"{protein_id}.pdb",
        protein_dir / "structure.pdb", 
        protein_dir / "protein.pdb",
        # XTC files (trajectory format)
        protein_dir / f"{protein_id}.xtc",
        protein_dir / "trajectory.xtc",
        # DCD files
        protein_dir / f"{protein_id}.dcd",
        protein_dir / "trajectory.dcd"
    ]
    
    # Find the first existing file
    for file_path in possible_files:
        if file_path.exists():
            try:
                return md.load(str(file_path))
            except Exception as e:
                print(f"    Warning: Could not load {file_path}: {e}")
                continue
    
    # If no specific file found, try to load any supported file in the directory
    supported_extensions = [".pdb", ".xtc", ".dcd", ".h5", ".netcdf"]
    for ext in supported_extensions:
        files = list(protein_dir.glob(f"*{ext}"))
        if files:
            try:
                return md.load(str(files[0]))
            except Exception as e:
                print(f"    Warning: Could not load {files[0]}: {e}")
                continue
    
    raise FileNotFoundError(f"No supported protein file found in {protein_dir}")


def compute_ca_distances(traj):
    """
    Compute CA-wise (alpha carbon) distances for a trajectory.
    Returns pairwise distances between all CA atoms.
    """
    # Get CA atom indices
    ca_indices = traj.topology.select("name CA")
    
    if len(ca_indices) == 0:
        raise ValueError("No CA atoms found in the trajectory")
    
    print(f"    Found {len(ca_indices)} CA atoms")
    
    # Create all pairs of CA atoms (upper triangular matrix)
    ca_pairs = []
    n_ca = len(ca_indices)
    for i in range(n_ca):
        for j in range(i + 1, n_ca):
            ca_pairs.append([ca_indices[i], ca_indices[j]])
    
    if len(ca_pairs) == 0:
        raise ValueError("No CA pairs could be formed")
    
    ca_pairs = np.array(ca_pairs)
    print(f"    Computing distances for {len(ca_pairs)} CA pairs")
    
    # Compute distances (result shape: n_frames x n_pairs)
    ca_distances = md.compute_distances(traj, ca_pairs)
    
    return ca_distances


def compute_tica_analysis(ca_distances, lags=[10, 100]):
    """
    Perform TICA analysis on CA distances with specified time lags.
    Based on the pattern from data-CLN025.ipynb notebook.
    """
    tica_models = {}
    
    print(f"    Performing TICA analysis with lags: {lags}")
    print(f"    Input data shape: {ca_distances.shape}")
    
    for lag in lags:
        try:
            # Create TICA model with specified lag and 2 dimensions
            # Following the pattern: pyemma.coordinates.tica(data, lag=lag, dim=2)
            tica_obj = pyemma.coordinates.tica(ca_distances, lag=lag, dim=2)
            tica_models[f"lag_{lag}"] = tica_obj
            print(f"    TICA lag {lag}: Created successfully")
            
        except Exception as e:
            print(f"    Warning: TICA analysis failed for lag {lag}: {e}")
            continue
        
    return tica_models


def process_single_protein(protein_id, data_dir="/home/shpark/prj-mlcv/lib/mdcath/data"):
    """
    Process a single protein: load data, extract xyz, compute CA distances, and perform TICA.
    """
    print(f"Processing protein: {protein_id}")
    
    protein_dir = Path(data_dir) / protein_id
    protein_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load protein trajectory
        traj = load_protein_data(protein_id, data_dir)
        print(f"  Loaded trajectory: {traj.n_frames} frames, {traj.n_atoms} atoms")
        
        # Extract xyz coordinates
        xyz_coords = traj.xyz  # Shape: (n_frames, n_atoms, 3)
        
        # Save xyz as pytorch tensor
        xyz_tensor = torch.from_numpy(xyz_coords.astype(np.float32))
        torch.save(xyz_tensor, protein_dir / "xyz.pt")
        print(f"  Saved xyz.pt: {xyz_coords.shape}")
        
        # Compute CA distances
        ca_distances = compute_ca_distances(traj)
        
        # Save CA distances as pytorch tensor
        ca_distances_tensor = torch.from_numpy(ca_distances.astype(np.float32))
        torch.save(ca_distances_tensor, protein_dir / "cad.pt")
        print(f"  Saved cad.pt: {ca_distances.shape}")
        
        # Check if we have enough frames for TICA analysis
        min_frames_needed = max(10, 100) + 1  # Need at least lag + 1 frames
        if traj.n_frames < min_frames_needed:
            print(f"  Warning: Not enough frames ({traj.n_frames}) for TICA analysis (need at least {min_frames_needed})")
            print(f"  Skipping TICA analysis for {protein_id}")
        else:
            # Perform TICA analysis
            tica_models = compute_tica_analysis(ca_distances, lags=[10, 100])
            
            # Save TICA models
            for model_name, tica_obj in tica_models.items():
                model_path = protein_dir / f"tica_model_{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(tica_obj, f)
                print(f"  Saved {model_path.name}")
            
        print(f"  Successfully processed {protein_id}")
        return True
        
    except Exception as e:
        print(f"  Error processing {protein_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main preprocessing function to process all proteins in the data directory.
    """
    # Try multiple possible data directory locations
    possible_data_dirs = [
        "/home/shpark/prj-mlcv/lib/mdcath/data",
        "mdcath/data",
        "./data"
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        if Path(dir_path).exists():
            data_dir = dir_path
            break
    
    if data_dir is None:
        print("No data directory found. Creating mdcath/data...")
        data_dir = "mdcath/data"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {data_dir}")
        print("Please place protein data folders in the created directory.")
        print("Expected structure: {data_dir}/{protein_id}/{protein_files}")
        return
    
    print(f"Using data directory: {data_dir}")
    data_path = Path(data_dir)
    
    # Get list of protein directories
    protein_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not protein_dirs:
        print(f"No protein directories found in {data_dir}")
        print(f"Expected structure: {data_dir}/{{protein_id}}/{{protein_files}}")
        
        # Create a sample directory structure for demonstration
        sample_dir = data_path / "1a0aA00"
        sample_dir.mkdir(exist_ok=True)
        print(f"Created sample directory: {sample_dir}")
        print("Place your protein structure files (PDB, XTC, DCD, etc.) in the protein directories.")
        return
    
    print(f"Found {len(protein_dirs)} protein directories")
    
    # Process each protein
    successful = 0
    failed = 0
    
    for protein_dir in tqdm(protein_dirs, desc="Processing proteins"):
        protein_id = protein_dir.name
        success = process_single_protein(protein_id, data_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nNote: Some proteins failed to process.")
        print("Common issues:")
        print("  - Missing or corrupted structure files")
        print("  - Structures without CA atoms")
        print("  - Insufficient frames for TICA analysis")
        print("  - Check the error messages above for details")


def test_single_protein(protein_id="1a0aA00", data_dir="mdcath/data"):
    """
    Test function to process a single protein for debugging.
    """
    print(f"Testing preprocessing for protein: {protein_id}")
    success = process_single_protein(protein_id, data_dir)
    
    if success:
        # Verify outputs
        protein_dir = Path(data_dir) / protein_id
        files_to_check = ["xyz.pt", "cad.pt"]
        
        print("\nVerifying outputs:")
        for filename in files_to_check:
            filepath = protein_dir / filename
            if filepath.exists():
                try:
                    data = torch.load(filepath)
                    print(f"  ✓ {filename}: shape {data.shape}")
                except Exception as e:
                    print(f"  ✗ {filename}: Error loading - {e}")
            else:
                print(f"  ✗ {filename}: File not found")
        
        # Check TICA models
        tica_files = list(protein_dir.glob("tica_model_*.pkl"))
        for tica_file in tica_files:
            try:
                with open(tica_file, 'rb') as f:
                    tica_obj = pickle.load(f)
                print(f"  ✓ {tica_file.name}: TICA model loaded successfully")
            except Exception as e:
                print(f"  ✗ {tica_file.name}: Error loading - {e}")
    
    return success


def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_modules = ['mdtraj', 'torch', 'pyemma', 'numpy', 'tqdm']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nInstall missing dependencies using:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    print("All required dependencies are available.")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess protein data for MDCath dataset")
    parser.add_argument("--data-dir", 
                       default="/home/shpark/prj-mlcv/lib/mdcath/data",
                       help="Path to protein data directory")
    parser.add_argument("--protein", 
                       help="Process only a specific protein ID")
    parser.add_argument("--test", 
                       action="store_true",
                       help="Run test with sample data")
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        exit(1)
    
    if args.test:
        # Run test mode
        print("Running in test mode...")
        success = test_single_protein("1a0aA00", "mdcath/data")
        exit(0 if success else 1)
    
    elif args.protein:
        # Process single protein
        print(f"Processing single protein: {args.protein}")
        success = process_single_protein(args.protein, args.data_dir)
        exit(0 if success else 1)
    
    else:
        # Run main processing
        main()