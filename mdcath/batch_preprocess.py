#!/usr/bin/env python3
"""
Batch preprocessing script for large protein datasets.
Supports parallel processing and resume functionality.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from preprocess import process_single_protein, check_dependencies


def process_protein_wrapper(args):
    """
    Wrapper function for multiprocessing.
    """
    protein_id, data_dir = args
    return protein_id, process_single_protein(protein_id, data_dir)


def save_progress(progress_file, completed_proteins, failed_proteins):
    """
    Save processing progress to a JSON file.
    """
    progress_data = {
        "completed": list(completed_proteins),
        "failed": list(failed_proteins),
        "timestamp": time.time()
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


def load_progress(progress_file):
    """
    Load processing progress from a JSON file.
    """
    if not Path(progress_file).exists():
        return set(), set()
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        completed = set(progress_data.get("completed", []))
        failed = set(progress_data.get("failed", []))
        return completed, failed
    
    except Exception as e:
        print(f"Warning: Could not load progress file {progress_file}: {e}")
        return set(), set()


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess protein data")
    parser.add_argument("--data-dir", 
                       default="/home/shpark/prj-mlcv/lib/mdcath/data",
                       help="Path to protein data directory")
    parser.add_argument("--workers", 
                       type=int, 
                       default=mp.cpu_count() // 2,
                       help="Number of parallel workers")
    parser.add_argument("--resume", 
                       action="store_true",
                       help="Resume from previous run")
    parser.add_argument("--progress-file",
                       default="processing_progress.json",
                       help="File to save processing progress")
    parser.add_argument("--proteins",
                       nargs="+",
                       help="Specific proteins to process (optional)")
    
    args = parser.parse_args()
    
    print("MDCath Batch Preprocessing")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running.")
        return 1
    
    # Check data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Data directory {args.data_dir} does not exist.")
        return 1
    
    # Get protein directories
    if args.proteins:
        protein_dirs = [data_path / protein_id for protein_id in args.proteins]
        protein_dirs = [d for d in protein_dirs if d.exists()]
    else:
        protein_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not protein_dirs:
        print("No protein directories found to process.")
        return 1
    
    protein_ids = [d.name for d in protein_dirs]
    print(f"Found {len(protein_ids)} proteins to process")
    
    # Load previous progress if resuming
    completed_proteins = set()
    failed_proteins = set()
    
    if args.resume:
        completed_proteins, failed_proteins = load_progress(args.progress_file)
        print(f"Resuming: {len(completed_proteins)} completed, {len(failed_proteins)} failed")
    
    # Filter out already processed proteins
    remaining_proteins = [pid for pid in protein_ids 
                         if pid not in completed_proteins and pid not in failed_proteins]
    
    if not remaining_proteins:
        print("All proteins have already been processed!")
        return 0
    
    print(f"Processing {len(remaining_proteins)} remaining proteins...")
    print(f"Using {args.workers} parallel workers")
    
    # Prepare arguments for parallel processing
    process_args = [(protein_id, args.data_dir) for protein_id in remaining_proteins]
    
    # Process proteins in parallel
    start_time = time.time()
    
    if args.workers == 1:
        # Sequential processing
        for protein_id in remaining_proteins:
            success = process_single_protein(protein_id, args.data_dir)
            if success:
                completed_proteins.add(protein_id)
            else:
                failed_proteins.add(protein_id)
            
            # Save progress periodically
            save_progress(args.progress_file, completed_proteins, failed_proteins)
    
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_protein = {
                executor.submit(process_protein_wrapper, (protein_id, args.data_dir)): protein_id
                for protein_id in remaining_proteins
            }
            
            # Process completed tasks
            for future in as_completed(future_to_protein):
                protein_id = future_to_protein[future]
                try:
                    result_protein_id, success = future.result()
                    if success:
                        completed_proteins.add(result_protein_id)
                        print(f"✓ Completed: {result_protein_id}")
                    else:
                        failed_proteins.add(result_protein_id)
                        print(f"✗ Failed: {result_protein_id}")
                
                except Exception as e:
                    failed_proteins.add(protein_id)
                    print(f"✗ Exception processing {protein_id}: {e}")
                
                # Save progress after each completion
                save_progress(args.progress_file, completed_proteins, failed_proteins)
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 40)
    print("Processing Summary:")
    print(f"  Total proteins: {len(protein_ids)}")
    print(f"  Successfully processed: {len(completed_proteins)}")
    print(f"  Failed: {len(failed_proteins)}")
    print(f"  Processing time: {total_time:.2f} seconds")
    
    if len(remaining_proteins) > 0:
        print(f"  Average time per protein: {total_time / len(remaining_proteins):.2f} seconds")
    
    # Save final progress
    save_progress(args.progress_file, completed_proteins, failed_proteins)
    
    return 0 if len(failed_proteins) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())