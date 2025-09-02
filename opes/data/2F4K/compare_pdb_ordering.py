#!/usr/bin/env python3
"""
Compare atom ordering between two PDB files and create a mapping.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict


def parse_pdb_atoms(pdb_file: Path) -> List[Tuple[int, str, str, str, int, float, float, float]]:
    """
    Parse PDB file and extract atom information.
    Returns list of tuples: (atom_number, atom_name, residue_name, chain, residue_number, x, y, z)
    """
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse PDB ATOM/HETATM record
                atom_number = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain = line[21:22].strip()
                residue_number = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                atoms.append((atom_number, atom_name, residue_name, chain, residue_number, x, y, z))
    
    return atoms


def create_atom_key(atom_info: Tuple) -> str:
    """Create a unique key for each atom based on residue and atom type."""
    atom_number, atom_name, residue_name, chain, residue_number, x, y, z = atom_info
    return f"{chain}_{residue_number}_{residue_name}_{atom_name}"


def find_mapping(maestro_atoms: List, pymol_atoms: List) -> List[int]:
    """
    Find mapping from maestro file to pymol file.
    Returns list where mapping[i] gives the pymol index for maestro atom i.
    """
    # Create dictionaries for fast lookup
    pymol_dict = {}
    for i, atom in enumerate(pymol_atoms):
        key = create_atom_key(atom)
        pymol_dict[key] = i
    
    mapping = []
    for i, maestro_atom in enumerate(maestro_atoms):
        maestro_key = create_atom_key(maestro_atom)
        
        if maestro_key in pymol_dict:
            # Found exact match
            pymol_index = pymol_dict[maestro_key]
            mapping.append(pymol_index + 1)  # +1 for 1-based indexing
        else:
            # Try to find by coordinates (in case of slight naming differences)
            maestro_coords = maestro_atom[5:8]  # x, y, z
            best_match = None
            min_distance = float('inf')
            
            for j, pymol_atom in enumerate(pymol_atoms):
                pymol_coords = pymol_atom[5:8]  # x, y, z
                distance = sum((a - b)**2 for a, b in zip(maestro_coords, pymol_coords))**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = j + 1  # +1 for 1-based indexing
            
            if min_distance < 0.01:  # Very small tolerance for coordinate matching
                mapping.append(best_match)
            else:
                print(f"Warning: No match found for maestro atom {i+1}: {maestro_key}")
                mapping.append(-1)  # Mark as not found
    
    return mapping


def analyze_differences(maestro_atoms: List, pymol_atoms: List, mapping: List[int]) -> None:
    """Analyze and report differences in atom ordering."""
    
    print(f"Total atoms in Maestro file: {len(maestro_atoms)}")
    print(f"Total atoms in PyMOL file: {len(pymol_atoms)}")
    print()
    
    # Count differences by residue type
    residue_diffs = {}
    atom_type_diffs = {}
    
    for i, (maestro_atom, pymol_idx) in enumerate(zip(maestro_atoms, mapping)):
        if pymol_idx == -1:
            continue
            
        maestro_key = create_atom_key(maestro_atom)
        pymol_atom = pymol_atoms[pymol_idx - 1]  # Convert back to 0-based
        pymol_key = create_atom_key(pymol_atom)
        
        # Check if ordering is different
        if i + 1 != pymol_idx:  # +1 for 1-based comparison
            residue_name = maestro_atom[2]
            atom_name = maestro_atom[1]
            
            if residue_name not in residue_diffs:
                residue_diffs[residue_name] = 0
            residue_diffs[residue_name] += 1
            
            if atom_name not in atom_type_diffs:
                atom_type_diffs[atom_name] = 0
            atom_type_diffs[atom_name] += 1
    
    print("Residue types with ordering differences:")
    for residue, count in sorted(residue_diffs.items()):
        print(f"  {residue}: {count} atoms")
    print()
    
    print("Atom types with ordering differences:")
    for atom_type, count in sorted(atom_type_diffs.items()):
        print(f"  {atom_type}: {count} atoms")
    print()


def main():
    # File paths
    maestro_file = Path("/home/shpark/prj-mlcv/lib/DESRES/data/2F4K/2f4k_from_maestro.pdb")
    pymol_file = Path("/home/shpark/prj-mlcv/lib/DESRES/data/2F4K/2f4k_from_pymol2.pdb")
    
    print("Parsing PDB files...")
    maestro_atoms = parse_pdb_atoms(maestro_file)
    pymol_atoms = parse_pdb_atoms(pymol_file)
    
    print("Creating mapping...")
    mapping = find_mapping(maestro_atoms, pymol_atoms)
    
    print("Analyzing differences...")
    analyze_differences(maestro_atoms, pymol_atoms, mapping)
    
    print("Mapping (Maestro -> PyMOL atom indices):")
    print("mapping =", mapping)
    print()
    
    # Show first 20 mappings as example
    print("First 20 atom mappings:")
    print("Maestro Index -> PyMOL Index (Residue, Atom)")
    for i in range(min(20, len(mapping))):
        maestro_atom = maestro_atoms[i]
        pymol_idx = mapping[i]
        if pymol_idx != -1:
            pymol_atom = pymol_atoms[pymol_idx - 1]
            print(f"{i+1:3d} -> {pymol_idx:3d}  ({maestro_atom[2]} {maestro_atom[1]} -> {pymol_atom[2]} {pymol_atom[1]})")
        else:
            print(f"{i+1:3d} -> ???  ({maestro_atom[2]} {maestro_atom[1]} -> NOT FOUND)")
    
    # Verify mapping correctness
    print("\nVerification - checking if coordinates match:")
    mismatches = 0
    for i, pymol_idx in enumerate(mapping[:10]):  # Check first 10
        if pymol_idx == -1:
            continue
        maestro_coords = maestro_atoms[i][5:8]
        pymol_coords = pymol_atoms[pymol_idx - 1][5:8]
        distance = sum((a - b)**2 for a, b in zip(maestro_coords, pymol_coords))**0.5
        if distance > 0.01:
            mismatches += 1
            print(f"  Mismatch at index {i+1}: distance = {distance:.6f}")
    
    if mismatches == 0:
        print("  All checked coordinates match perfectly!")
    else:
        print(f"  Found {mismatches} coordinate mismatches")


if __name__ == "__main__":
    main()
