import torch
import pickle
import numpy as np
import mdtraj as md
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations
from pathlib import Path
import argparse
import sys


def plot_tica_scatter(simulation_traj, molecule, method, output_dir=None):
    """
    Generate scatter plot over the original TICA plot
    
    Args:
        simulation_traj: MDTraj trajectory object
        molecule: Molecule name (e.g., 'cln025')
        output_dir: Directory to save plots (optional)
    """
    print("Generating TICA scatter plot...")
    
    # Load TICA model
    tica_model_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule.upper()}/{molecule.upper()}_tica_model_switch_lag10.pkl"
    try:
        with open(tica_model_path, 'rb') as f:
            tica_model = pickle.load(f)
        print(f"Loaded TICA model: {tica_model}")
    except FileNotFoundError:
        print(f"Error: TICA model not found at {tica_model_path}")
        return
    
    # Compute contacts for simulation trajectory
    ca_resid_pair = np.array(
        [(a.index, b.index) for a, b in combinations(list(simulation_traj.topology.residues), 2)]
    )
    ca_pair_contacts, resid_pairs = md.compute_contacts(
        simulation_traj, scheme="ca", contacts=ca_resid_pair, periodic=False
    )
    
    # Apply switching function and transform to TICA coordinates
    ca_pair_contacts_switch = (1 - np.power(ca_pair_contacts / 0.8, 6)) / (1 - np.power(ca_pair_contacts / 0.8, 12))
    simulation_tica_coord = tica_model.transform(ca_pair_contacts_switch)
    simulation_tica_x = simulation_tica_coord[:, 0]
    simulation_tica_y = simulation_tica_coord[:, 1]
    
    # Load full dataset for background
    cad_full_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule.upper()}-all/cad-switch.pt"
    try:
        cad_full = torch.load(cad_full_path)
        tica_coord = tica_model.transform(cad_full.numpy())
        print(f"Loaded full dataset: {cad_full.shape}")
    except FileNotFoundError:
        print(f"Warning: Full dataset not found at {cad_full_path}, creating plot without background")
        tica_coord = None
    
    # Create plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    if tica_coord is not None:
        ax.hist2d(tica_coord[:, 0], tica_coord[:, 1], bins=100, norm=LogNorm(), alpha=0.3)
    
    ax.scatter(simulation_tica_x, simulation_tica_y, c='red', s=20, alpha=0.7, label='Simulation trajectory')
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title("TICA Projection with Simulation Trajectory")
    ax.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / f"opes_{method}_tica_scatter.png", dpi=300, bbox_inches='tight')
        print(f"TICA scatter plot saved to {output_dir}/opes_{method}_tica_scatter.png")
    
    plt.show()
    plt.close()


def plot_cv_over_time(simulation_traj, molecule, method, output_dir=None):
    """
    Generate CV over time plot
    
    Args:
        simulation_traj: MDTraj trajectory object
        molecule: Molecule name (e.g., 'cln025')
        output_dir: Directory to save plots (optional)
    """
    print("Generating CV over time plot...")
    
    # Compute contacts
    ca_resid_pair = np.array(
        [(a.index, b.index) for a, b in combinations(list(simulation_traj.topology.residues), 2)]
    )
    ca_pair_contacts, resid_pairs = md.compute_contacts(
        simulation_traj, scheme="ca", contacts=ca_resid_pair, periodic=False
    )
    
    # Load CV model
    model_path = f"/home/shpark/prj-mlcv/lib/bioemu/model/_baseline_/tda-{molecule.upper()}-jit.pt"
    try:
        model = torch.jit.load(model_path)
        cv = model(torch.from_numpy(ca_pair_contacts))
        print(f"CV shape: {cv.shape}")
    except FileNotFoundError:
        print(f"Error: CV model not found at {model_path}")
        return
    
    # Create plot
    n_frames, n_cvs = cv.shape
    time = np.arange(n_frames)
    cv_np = cv.detach().numpy()
    
    plt.figure(figsize=(10, 6))
    for i in range(n_cvs):
        plt.plot(time, cv_np[:, i], label=f"CV {i}", alpha=0.8, linewidth=2)
    
    plt.xlabel("Time (frames)")
    plt.ylabel("CV Value")
    plt.title("Time Evolution of Collective Variables")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / f"opes_{method}_cv_over_time.png", dpi=300, bbox_inches='tight')
        print(f"CV over time plot saved to {output_dir}/opes_{method}_cv_over_time.png")
    
    plt.show()
    plt.close()


def plot_rmsd_over_time(simulation_traj, molecule, method, output_dir=None):
    """
    Generate RMSD over time plot
    
    Args:
        simulation_traj: MDTraj trajectory object
        molecule: Molecule name (e.g., 'cln025')
        output_dir: Directory to save plots (optional)
    """
    print("Generating RMSD over time plot...")
    
    # Load reference structure
    reference_traj_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule.upper()}/folded.pdb"
    try:
        reference_traj = md.load_pdb(reference_traj_path)
        print(f"Loaded reference structure from {reference_traj_path}")
    except FileNotFoundError:
        print(f"Error: Reference structure not found at {reference_traj_path}")
        return
    
    # Compute RMSD
    rmsd = md.rmsd(simulation_traj, reference_traj)
    print(f"RMSD shape: {rmsd.shape}")
    
    # Create plot
    time = np.arange(rmsd.shape[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, rmsd, label="RMSD", alpha=0.8, linewidth=2, color='green')
    plt.xlabel("Time (frames)")
    plt.ylabel("RMSD (nm)")
    plt.title("Time Evolution of RMSD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / f"opes_{method}_rmsd_over_time.png", dpi=300, bbox_inches='tight')
        print(f"RMSD over time plot saved to {output_dir}/opes_{method}_rmsd_over_time.png")
    
    plt.show()
    plt.close()


def analyze_trajectory(trajectory_path, topology_path, molecule, method, output_dir=None):
    """
    Main analysis function that generates all three plots
    
    Args:
        trajectory_path: Path to trajectory file (.xtc)
        topology_path: Path to topology file (.pdb)
        molecule: Molecule name (e.g., 'cln025')
        output_dir: Directory to save plots (optional)
    """
    print(f"Loading trajectory from {trajectory_path}")
    print(f"Using topology from {topology_path}")
    
    # Load trajectory
    try:
        simulation_traj = md.load_xtc(trajectory_path, top=topology_path)
        print(f"Loaded trajectory with {len(simulation_traj)} frames")
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")
    
    # Generate all plots
    print("\n" + "="*50)
    plot_tica_scatter(simulation_traj, molecule, method, output_dir)
    
    print("\n" + "="*50)
    plot_cv_over_time(simulation_traj, molecule, method, output_dir)
    
    print("\n" + "="*50)
    plot_rmsd_over_time(simulation_traj, molecule, method, output_dir)
    
    print("\n" + "="*50)
    print("Analysis completed!")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Analyze OPES simulation trajectories")
    parser.add_argument("method", help="Method name (e.g., tae, tica, vde)")
    parser.add_argument("trajectory", help="Path to trajectory file (.xtc)")
    parser.add_argument("topology", help="Path to topology file (.pdb)")
    parser.add_argument("molecule", help="Molecule name (e.g., cln025)")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    
    args = parser.parse_args()
    
    analyze_trajectory(args.trajectory, args.topology, args.molecule, args.method, args.output)


if __name__ == "__main__":
    main()
