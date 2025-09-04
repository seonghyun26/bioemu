import torch
import numpy as np
import argparse

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from omegaconf import OmegaConf
from pathlib import Path
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="tda")
    parser.add_argument("--molecule", type=str, default="2JOF")
    args = parser.parse_args()
    
    device = "cuda:0"
    method = args.method
    molecule = args.molecule
    date = f"0814_073849"
    
    batch_size_eval = 10000
    mlcv_model = torch.jit.load(f"/home/shpark/prj-mlcv/lib/bioemu/opes/model/_baseline_/{method}-{molecule}-jit.pt", map_location=device)
    # mlcv_model = torch.jit.load(f"/home/shpark/prj-mlcv/lib/bioemu/model/{date}/mlcv_model-jit.pt", map_location=device)
    mlcv_model.eval()
    mlcv_model.to(device)
    
    mlcv_save_path = f"/home/shpark/prj-mlcv/lib/bioemu/opes/dataset/{molecule.upper()}-all/{method}_mlcv.npy"
    if os.path.exists(mlcv_save_path):
        print(f"> CV values already computed at {mlcv_save_path}")
        exit()
    
    projection_data_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-cad.pt"
    projection_data = torch.load(projection_data_path).to(device)
    eval_loader = DataLoader(
        TensorDataset(projection_data),
        batch_size=batch_size_eval,
        shuffle=False
    )
    
    with torch.no_grad():
        sample_batch = next(iter(eval_loader))[0]
        sample_output = mlcv_model(sample_batch)
        output_dim = sample_output.shape[1]
    cv_batches = torch.zeros((len(projection_data), output_dim)).to(projection_data.device)
    print(f"CV shape: {cv_batches.shape}")
    
    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(tqdm(
            eval_loader,
            desc="Computing CV values",
            total=len(eval_loader),
            leave=False,
        )):
            batch_cv = mlcv_model(batch_data)
            start_idx = batch_idx * batch_size_eval
            end_idx = start_idx + batch_cv.shape[0]  # Handle last batch size correctly
            cv_batches[start_idx:end_idx] = batch_cv
    
    cv = cv_batches.detach().cpu().numpy()
    print(f"\nMethod: {method}, Molecule: {molecule}")
    print(f"CV shape: {cv.shape}")
    print(f"CV range: [{cv.min():.6f}, {cv.max():.6f}]")
    print(f"CV mean: {cv.mean():.6f}, std: {cv.std():.6f}")
    
    np.save(mlcv_save_path, cv)
    print(f"Saved CV values to {mlcv_save_path}")