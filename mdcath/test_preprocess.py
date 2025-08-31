#!/usr/bin/env python3
"""
Test script for the protein preprocessing pipeline.
This script demonstrates how to use the preprocessing functions.
"""

import sys
from pathlib import Path

# Add current directory to path to import preprocess module
sys.path.append(str(Path(__file__).parent))

from preprocess import check_dependencies, test_single_protein, process_single_protein


def create_sample_pdb():
    """
    Create a minimal sample PDB file for testing.
    This creates a simple protein structure with CA atoms.
    """
    sample_pdb_content = """HEADER    SAMPLE PROTEIN                          01-JAN-70   1A0A
ATOM      1  N   ALA A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  17.889  27.069  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.618  17.257  27.147  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.534  16.018  27.227  1.00 20.00           O  
ATOM      5  CB  ALA A   1      19.267  18.537  25.709  1.00 20.00           C  
ATOM      6  N   VAL A   2      16.540  18.035  27.117  1.00 20.00           N  
ATOM      7  CA  VAL A   2      15.166  17.535  27.184  1.00 20.00           C  
ATOM      8  C   VAL A   2      14.096  18.609  27.011  1.00 20.00           C  
ATOM      9  O   VAL A   2      14.092  19.339  26.024  1.00 20.00           O  
ATOM     10  CB  VAL A   2      14.881  16.457  26.113  1.00 20.00           C  
ATOM     11  CG1 VAL A   2      13.447  15.942  26.162  1.00 20.00           C  
ATOM     12  CG2 VAL A   2      15.875  15.318  26.235  1.00 20.00           C  
ATOM     13  N   GLY A   3      13.168  18.654  27.961  1.00 20.00           N  
ATOM     14  CA  GLY A   3      12.099  19.647  27.887  1.00 20.00           C  
ATOM     15  C   GLY A   3      10.716  19.058  27.682  1.00 20.00           C  
ATOM     16  O   GLY A   3      10.673  17.847  27.478  1.00 20.00           O  
ATOM     17  N   LEU A   4       9.617  19.807  27.742  1.00 20.00           N  
ATOM     18  CA  LEU A   4       8.279  19.316  27.556  1.00 20.00           C  
ATOM     19  C   LEU A   4       7.174  20.369  27.413  1.00 20.00           C  
ATOM     20  O   LEU A   4       7.211  21.364  28.134  1.00 20.00           O  
ATOM     21  CB  LEU A   4       7.912  18.385  28.721  1.00 20.00           C  
ATOM     22  CG  LEU A   4       8.807  17.166  28.925  1.00 20.00           C  
ATOM     23  CD1 LEU A   4       8.434  16.369  30.163  1.00 20.00           C  
ATOM     24  CD2 LEU A   4       8.759  16.284  27.687  1.00 20.00           C  
END
"""
    
    # Create sample directory and PDB file
    sample_dir = Path("mdcath/data/1a0aA00")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_file = sample_dir / "1a0aA00.pdb"
    with open(pdb_file, 'w') as f:
        f.write(sample_pdb_content)
    
    print(f"Created sample PDB file: {pdb_file}")
    return pdb_file


def main():
    """
    Main test function.
    """
    print("MDCath Preprocessing Test")
    print("=" * 40)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("Please install missing dependencies before running the test.")
        return
    
    # Create sample data if needed
    print("\n2. Creating sample data...")
    try:
        sample_pdb = create_sample_pdb()
        print(f"Sample PDB created successfully")
    except Exception as e:
        print(f"Error creating sample PDB: {e}")
        return
    
    # Test preprocessing
    print("\n3. Testing preprocessing...")
    try:
        success = test_single_protein("1a0aA00", "mdcath/data")
        if success:
            print("\n✓ Preprocessing test completed successfully!")
        else:
            print("\n✗ Preprocessing test failed!")
    except Exception as e:
        print(f"\n✗ Preprocessing test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()