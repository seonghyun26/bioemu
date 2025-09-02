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
from scipy import stats
from sklearn.neighbors import KernelDensity


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


def compute_free_energy_surface(simulation_traj, molecule, method, output_dir=None, cv_dims=2, bins=50):
    """
    Compute free energy surface from trajectory using TICA coordinates
    
    Args:
        simulation_traj: MDTraj trajectory object
        molecule: Molecule name (e.g., 'cln025')
        method: Method name for saving
        output_dir: Directory to save plots and data (optional)
        cv_dims: Number of CV dimensions to use (default: 2)
        bins: Number of bins for histogram (default: 50)
    
    Returns:
        dict: Dictionary containing free energy data
    """
    print("Computing free energy surface...")
    
    # Load TICA model
    tica_model_path = f"/home/shpark/prj-mlcv/lib/DESRES/data/{molecule.upper()}/{molecule.upper()}_tica_model_switch_lag10.pkl"
    try:
        with open(tica_model_path, 'rb') as f:
            tica_model = pickle.load(f)
        print(f"Loaded TICA model: {tica_model}")
    except FileNotFoundError:
        print(f"Error: TICA model not found at {tica_model_path}")
        return None
    
    # Compute contacts for simulation trajectory
    ca_resid_pair = np.array(
        [(a.index, b.index) for a, b in combinations(list(simulation_traj.topology.residues), 2)]
    )
    ca_pair_contacts, resid_pairs = md.compute_contacts(
        simulation_traj, scheme="ca", contacts=ca_resid_pair, periodic=False
    )
    
    # Apply switching function and transform to TICA coordinates
    ca_pair_contacts_switch = (1 - np.power(ca_pair_contacts / 0.8, 6)) / (1 - np.power(ca_pair_contacts / 0.8, 12))
    tica_coords = tica_model.transform(ca_pair_contacts_switch)
    
    # Use specified number of CV dimensions
    tica_coords = tica_coords[:, :cv_dims]
    print(f"Using {cv_dims} TICA dimensions, trajectory shape: {tica_coords.shape}")
    
    # Compute probability distribution
    if cv_dims == 1:
        # 1D case
        hist, bin_edges = np.histogram(tica_coords[:, 0], bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute free energy (F = -kT * ln(P))
        kT = 2.49  # kJ/mol at 300K
        prob = hist / np.sum(hist)  # Normalize
        prob[prob == 0] = np.min(prob[prob > 0])  # Avoid log(0)
        free_energy = -kT * np.log(prob)
        free_energy = free_energy - np.min(free_energy)  # Set minimum to 0
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, free_energy, 'b-', linewidth=2, label='Free Energy')
        plt.xlabel('TICA 1')
        plt.ylabel('Free Energy (kJ/mol)')
        plt.title(f'1D Free Energy Profile - {method}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save results
        results = {
            'method': method,
            'molecule': molecule,
            'cv_dims': cv_dims,
            'bins': bins,
            'tica_coords': tica_coords,
            'bin_centers': bin_centers,
            'free_energy': free_energy,
            'probability': prob
        }
        
    elif cv_dims == 2:
        # 2D case
        hist, x_edges, y_edges = np.histogram2d(
            tica_coords[:, 0], tica_coords[:, 1], 
            bins=bins, density=True
        )
        
        # Create coordinate grids
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # Compute free energy surface
        kT = 2.49  # kJ/mol at 300K
        prob = hist.T / np.sum(hist)  # Normalize and transpose for correct orientation
        prob[prob == 0] = np.min(prob[prob > 0])  # Avoid log(0)
        free_energy = -kT * np.log(prob)
        free_energy = free_energy - np.min(free_energy)  # Set minimum to 0
        
        # Create contour plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Probability surface
        im1 = ax1.contourf(X, Y, prob, levels=20, cmap='viridis')
        ax1.set_xlabel('TICA 1')
        ax1.set_ylabel('TICA 2')
        ax1.set_title(f'Probability Distribution - {method}')
        plt.colorbar(im1, ax=ax1, label='Probability')
        
        # Free energy surface
        levels = np.linspace(0, np.min([np.max(free_energy), 25]), 15)  # Cap at 25 kJ/mol
        im2 = ax2.contourf(X, Y, free_energy, levels=levels, cmap='coolwarm')
        contours = ax2.contour(X, Y, free_energy, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        ax2.set_xlabel('TICA 1')
        ax2.set_ylabel('TICA 2')
        ax2.set_title(f'Free Energy Surface - {method}')
        plt.colorbar(im2, ax=ax2, label='Free Energy (kJ/mol)')
        
        plt.tight_layout()
        
        # Save results
        results = {
            'method': method,
            'molecule': molecule,
            'cv_dims': cv_dims,
            'bins': bins,
            'tica_coords': tica_coords,
            'x_centers': x_centers,
            'y_centers': y_centers,
            'X': X,
            'Y': Y,
            'free_energy': free_energy,
            'probability': prob
        }
        
    else:
        print(f"Error: {cv_dims}D free energy computation not implemented")
        return None
    
    # Save plot
    if output_dir:
        plot_path = Path(output_dir) / f"opes_{method}_free_energy_{cv_dims}d.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Free energy plot saved to {plot_path}")
        
        # Save data in PyTorch format
        data_path = Path(output_dir) / f"opes_{method}_free_energy_{cv_dims}d.pt"
        torch.save(results, data_path)
        print(f"Free energy data saved to {data_path}")
    
    plt.show()
    plt.close()
    
    return results


def compute_free_energy_from_colvar(colvar_path, output_dir=None, method="unknown"):
    """
    Compute free energy from PLUMED COLVAR file
    
    Args:
        colvar_path: Path to COLVAR file
        output_dir: Directory to save plots and data (optional)
        method: Method name for saving
    
    Returns:
        dict: Dictionary containing free energy data
    """
    print(f"Computing free energy from COLVAR file: {colvar_path}")
    
    try:
        # Load COLVAR data
        data = np.loadtxt(colvar_path, comments='#')
        if len(data) == 0:
            print("Error: COLVAR file is empty")
            return None
            
        # Read header to get column names
        with open(colvar_path, 'r') as f:
            header = f.readline().strip()
            keys = header.split()[2:]  # Skip '#' and 'FIELDS'
        
        print(f"COLVAR columns: {keys}")
        
        # Get CV data (assuming first CV after time)
        time_idx = keys.index('time') if 'time' in keys else 0
        cv_columns = [i for i, key in enumerate(keys) if key not in ['time']]
        
        if len(cv_columns) == 0:
            print("Error: No CV columns found in COLVAR file")
            return None
        
        times = data[:, time_idx] / 1000  # Convert ps to ns
        cv_data = data[:, cv_columns[0]]  # Use first CV
        
        print(f"Using CV column: {keys[cv_columns[0]]}")
        print(f"CV data shape: {cv_data.shape}")
        
        # Compute free energy profile
        bins = 50
        hist, bin_edges = np.histogram(cv_data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute free energy (F = -kT * ln(P))
        kT = 2.49  # kJ/mol at 300K
        prob = hist / np.sum(hist)  # Normalize
        prob[prob == 0] = np.min(prob[prob > 0])  # Avoid log(0)
        free_energy = -kT * np.log(prob)
        free_energy = free_energy - np.min(free_energy)  # Set minimum to 0
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # CV over time
        ax1.plot(times, cv_data, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel(f'{keys[cv_columns[0]]}')
        ax1.set_title(f'CV Evolution - {method}')
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(cv_data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel(f'{keys[cv_columns[0]]}')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'CV Distribution - {method}')
        ax2.grid(True, alpha=0.3)
        
        # Free energy profile
        ax3.plot(bin_centers, free_energy, 'r-', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel(f'{keys[cv_columns[0]]}')
        ax3.set_ylabel('Free Energy (kJ/mol)')
        ax3.set_title(f'Free Energy Profile - {method}')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save results
        results = {
            'method': method,
            'colvar_path': str(colvar_path),
            'cv_name': keys[cv_columns[0]],
            'times': times,
            'cv_data': cv_data,
            'bin_centers': bin_centers,
            'free_energy': free_energy,
            'probability': prob,
            'histogram': hist
        }
        
        # Save plot and data
        if output_dir:
            plot_path = Path(output_dir) / f"opes_{method}_colvar_free_energy.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"COLVAR free energy plot saved to {plot_path}")
            
            # Save data in PyTorch format
            data_path = Path(output_dir) / f"opes_{method}_colvar_free_energy.pt"
            torch.save(results, data_path)
            print(f"COLVAR free energy data saved to {data_path}")
        
        plt.show()
        plt.close()
        
        return results
        
    except Exception as e:
        print(f"Error processing COLVAR file: {e}")
        return None


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
    # Compute free energy surfaces (both 1D and 2D)
    free_energy_1d = compute_free_energy_surface(simulation_traj, molecule, method, output_dir, cv_dims=1)
    
    print("\n" + "="*50)
    free_energy_2d = compute_free_energy_surface(simulation_traj, molecule, method, output_dir, cv_dims=2)
    
    print("\n" + "="*50)
    print("Analysis completed!")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Analyze OPES simulation trajectories")
    subparsers = parser.add_subparsers(dest='mode', help='Analysis mode')
    
    # Trajectory analysis
    traj_parser = subparsers.add_parser('trajectory', help='Analyze trajectory files')
    traj_parser.add_argument("method", help="Method name (e.g., tae, tica, vde)")
    traj_parser.add_argument("trajectory", help="Path to trajectory file (.xtc)")
    traj_parser.add_argument("topology", help="Path to topology file (.pdb)")
    traj_parser.add_argument("molecule", help="Molecule name (e.g., cln025)")
    traj_parser.add_argument("--output", "-o", help="Output directory for plots")
    
    # COLVAR analysis
    colvar_parser = subparsers.add_parser('colvar', help='Analyze COLVAR files')
    colvar_parser.add_argument("method", help="Method name (e.g., tae, tica, vde)")
    colvar_parser.add_argument("colvar", help="Path to COLVAR file")
    colvar_parser.add_argument("--output", "-o", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if args.mode == 'trajectory':
        analyze_trajectory(args.trajectory, args.topology, args.molecule, args.method, args.output)
    elif args.mode == 'colvar':
        compute_free_energy_from_colvar(args.colvar, args.output, args.method)
    else:
        # Backward compatibility - assume trajectory analysis if no subcommand
        if len(sys.argv) >= 5:
            method, trajectory, topology, molecule = sys.argv[1:5]
            output = None
            if len(sys.argv) > 5 and (sys.argv[5] == '--output' or sys.argv[5] == '-o'):
                output = sys.argv[6] if len(sys.argv) > 6 else None
            analyze_trajectory(trajectory, topology, molecule, method, output)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
