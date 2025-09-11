"""
Script to generate system-specific MDP files with customizable temperature.

This script takes the template MDP files (_nvt.mdp, _npt.mdp, _md.mdp) and creates
system-specific versions with the temperature changed to the specified value.

Usage:
    python generate_mdp.py <system_name> <temperature> [output_dir]

Example:
    python generate_mdp.py 1fme 325
    python generate_mdp.py cln025 300 ./output/
"""

import os
import sys
import argparse
from pathlib import Path


def update_temperature_in_mdp(content, temperature):
    """
    Update temperature-related parameters in MDP content.
    
    Args:
        content (str): MDP file content
        temperature (float): Target temperature in Kelvin
    
    Returns:
        str: Updated MDP content
    """
    lines = content.split('\n')
    updated_lines = []
    
    for line in lines:
        # Update ref_t parameter (reference temperature for thermostat)
        if line.strip().startswith('ref_t'):
            # Extract existing values and replace with new temperature
            parts = line.split('=')
            if len(parts) == 2:
                # Keep the same format but update temperature values
                updated_lines.append(f"ref_t                   = {temperature}     {temperature}           ; reference temperature, one for each group, in K")
            else:
                updated_lines.append(line)
        # Update gen-temp parameter (for velocity generation in NVT)
        elif line.strip().startswith('gen-temp'):
            parts = line.split('=')
            if len(parts) == 2:
                updated_lines.append(f"gen-temp                = {temperature}")
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


def update_title_in_mdp(content, system_name, mdp_type):
    """
    Update the title in MDP content to reflect the system and type.
    
    Args:
        content (str): MDP file content
        system_name (str): Name of the system
        mdp_type (str): Type of MDP (nvt, npt, md)
    
    Returns:
        str: Updated MDP content
    """
    lines = content.split('\n')
    updated_lines = []
    
    for line in lines:
        if line.strip().startswith('title'):
            # Update title to reflect system and type
            if mdp_type.upper() == 'NVT':
                updated_lines.append(f"title                   = {system_name.upper()} NVT equilibration")
            elif mdp_type.upper() == 'NPT':
                updated_lines.append(f"title                   = {system_name.upper()} NPT equilibration")
            elif mdp_type.upper() == 'MD':
                updated_lines.append(f"title                   = {system_name.upper()} MD production")
            else:
                updated_lines.append(f"title                   = {system_name.upper()} {mdp_type.upper()} simulation")
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


def generate_mdp_file(template_path, output_path, system_name, temperature, mdp_type):
    """
    Generate a system-specific MDP file from template.
    
    Args:
        template_path (str): Path to template MDP file
        output_path (str): Path for output MDP file
        system_name (str): Name of the system
        temperature (float): Target temperature in Kelvin
        mdp_type (str): Type of MDP (nvt, npt, md)
    """
    try:
        # Read template file
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Update temperature parameters
        content = update_temperature_in_mdp(content, temperature)
        
        # Update title
        content = update_title_in_mdp(content, system_name, mdp_type)
        
        # Write output file
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"Generated: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Template file not found: {template_path}")
    except Exception as e:
        print(f"Error generating {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate system-specific MDP files with customizable temperature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_mdp.py 1fme 325
  python generate_mdp.py cln025 300 ./output/
  python generate_mdp.py myprotein 310 /path/to/output/
        """
    )
    
    parser.add_argument('system_name', help='Name of the system (e.g., 1fme, cln025)')
    parser.add_argument('temperature', type=int, help='Target temperature in Kelvin')
    parser.add_argument('output_dir', nargs='?', default='.', 
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define template files
    templates = {
        'nvt': script_dir / '_nvt.mdp',
        'npt': script_dir / '_npt.mdp', 
        'md': script_dir / '_md.mdp'
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(f"../{args.system_name.upper()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate MDP files for each type
    for mdp_type, template_path in templates.items():
        if template_path.exists():
            output_filename = f"{args.system_name}_{mdp_type}.mdp"
            output_path = output_dir / output_filename
            
            generate_mdp_file(
                str(template_path),
                str(output_path),
                args.system_name,
                args.temperature,
                mdp_type
            )
        else:
            print(f"Warning: Template file not found: {template_path}")
    
    print(f"\nGenerated MDP files for system '{args.system_name}' at temperature {args.temperature}K")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
