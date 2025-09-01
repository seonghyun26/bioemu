#!/usr/bin/env python3
"""
Simple analysis of PDB atom ordering differences.
"""

def extract_atoms_from_content(content_lines):
    """Extract atom information from PDB content."""
    atoms = []
    for line in content_lines:
        if line.strip().startswith('ATOM') or line.strip().startswith('HETATM'):
            # Remove the line number prefix if present
            if '|' in line:
                line = line.split('|', 1)[1]
            
            if len(line) >= 54:  # Ensure line is long enough
                atom_num = line[6:11].strip()
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21:22].strip()
                res_num = line[22:26].strip()
                x = line[30:38].strip()
                y = line[38:46].strip()
                z = line[46:54].strip()
                
                try:
                    atoms.append({
                        'atom_num': int(atom_num),
                        'atom_name': atom_name,
                        'res_name': res_name,
                        'chain': chain,
                        'res_num': int(res_num),
                        'coords': (float(x), float(y), float(z))
                    })
                except ValueError:
                    continue
    return atoms

# Read maestro file content
maestro_content = """REMARK   4      COMPLIES WITH FORMAT V. 3.0, 1-DEC-2006
REMARK 888
REMARK 888 WRITTEN BY MAESTRO (A PRODUCT OF SCHRODINGER, LLC)
CRYST1   52.261   52.261   52.261  90.00  90.00  90.00 P 1           1
ATOM      1  N   LEU X  42      -3.676 -13.752   3.950  1.00  0.00           N  
ATOM      2  CA  LEU X  42      -3.823 -14.505   5.186  1.00  0.00           C  
ATOM      3  C   LEU X  42      -5.087 -14.004   5.956  1.00  0.00           C  
ATOM      4  O   LEU X  42      -6.010 -14.780   6.124  1.00  0.00           O  
ATOM      5  CB  LEU X  42      -2.577 -14.530   6.102  1.00  0.00           C  
ATOM      6  CG  LEU X  42      -1.193 -15.031   5.538  1.00  0.00           C  
ATOM      7  CD1 LEU X  42       0.039 -14.849   6.402  1.00  0.00           C  
ATOM      8  CD2 LEU X  42      -1.224 -16.532   5.195  1.00  0.00           C""".split('\n')

# Read pymol file content  
pymol_content = """CRYST1   52.261   52.261   52.261  90.00  90.00  90.00               0
ATOM      1  N   LEU X  42      -3.676 -13.752   3.950  1.00  0.00           N  
ATOM      2  CA  LEU X  42      -3.823 -14.505   5.186  1.00  0.00           C  
ATOM      3  CB  LEU X  42      -2.577 -14.530   6.102  1.00  0.00           C  
ATOM      4  CG  LEU X  42      -1.193 -15.031   5.538  1.00  0.00           C  
ATOM      5  CD1 LEU X  42       0.039 -14.849   6.402  1.00  0.00           C  
ATOM      6  CD2 LEU X  42      -1.224 -16.532   5.195  1.00  0.00           C  
ATOM      7  C   LEU X  42      -5.087 -14.004   5.956  1.00  0.00           C  
ATOM      8  O   LEU X  42      -6.010 -14.780   6.124  1.00  0.00           O""".split('\n')

print("Analyzing first few atoms from each file:")
print("\nMaestro file atoms:")
maestro_atoms = extract_atoms_from_content(maestro_content)
for i, atom in enumerate(maestro_atoms[:8]):
    print(f"{i+1:2d}: {atom['atom_name']:4s} {atom['res_name']} {atom['res_num']}")

print("\nPyMOL file atoms:")
pymol_atoms = extract_atoms_from_content(pymol_content)
for i, atom in enumerate(pymol_atoms[:8]):
    print(f"{i+1:2d}: {atom['atom_name']:4s} {atom['res_name']} {atom['res_num']}")

print("\nMapping analysis:")
print("Looking at the ordering difference...")

# Create mapping for first 8 atoms
maestro_keys = [(atom['res_num'], atom['res_name'], atom['atom_name']) for atom in maestro_atoms[:8]]
pymol_keys = [(atom['res_num'], atom['res_name'], atom['atom_name']) for atom in pymol_atoms[:8]]

print("\nMaestro order:", [f"{res}{name}" for res, resname, name in maestro_keys])
print("PyMOL order:  ", [f"{res}{name}" for res, resname, name in pymol_keys])

# Find mapping
mapping = []
for maestro_key in maestro_keys:
    try:
        pymol_index = pymol_keys.index(maestro_key)
        mapping.append(pymol_index + 1)  # 1-based indexing
    except ValueError:
        mapping.append(-1)

print(f"\nMapping for first 8 atoms: {mapping}")
print("\nThis means:")
for i, pymol_idx in enumerate(mapping):
    if pymol_idx != -1:
        maestro_atom = maestro_atoms[i]
        pymol_atom = pymol_atoms[pymol_idx - 1]
        print(f"Maestro atom {i+1} ({maestro_atom['atom_name']}) comes from PyMOL atom {pymol_idx} ({pymol_atom['atom_name']})")
