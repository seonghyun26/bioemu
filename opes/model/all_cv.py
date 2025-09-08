import torch
import numpy as np
import argparse
import wandb

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt

from omegaconf import OmegaConf
from pathlib import Path
import os


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--method", type=str, default="tda")
    # parser.add_argument("--molecule", type=str, default="2JOF")
    # args = parser.parse_args()
    
    # molecule_list = ["CLN025","2JOF","2F4K","1FME","GTT","NTL9"]
    # method_list = ["tda","tica","tae","vde"]
    molecule_list = ["2JOF"]
    method_list = ["ours"]
    
    device = "cuda:0"
    
    for molecule in molecule_list:
        for method in method_list:
            print(f"\nMethod: {method}, Molecule: {molecule}")
            batch_size_eval = 10000
            if method != "ours":
                mlcv_model = torch.jit.load(f"/home/shpark/prj-mlcv/lib/bioemu/opes/model/_baseline_/{method}-{molecule}-jit.pt", map_location=device)
                mlcv_model.eval()
            else:
                model_ckpt_mapping = {
                    "2JOF": "0814_073849",
                    # "2F4K": "0819_173704",
                    "1FME": "0904_160804",
                    "NTL9": "0905_054344",
                    "GTT": "0905_160702",
                }
                mlcv_model = torch.jit.load(f"/home/shpark/prj-mlcv/lib/bioemu/opes/model/{model_ckpt_mapping[molecule]}-{molecule}-jit.pt", map_location=device)
                mlcv_model.eval()
            
            dataset_dir = Path(f"/home/shpark/prj-mlcv/lib/bioemu/opes/{molecule.upper()}")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            mlcv_save_path = dataset_dir / f"{method}_mlcv.npy"
            
            if os.path.exists(mlcv_save_path):
                print(f"> CV values already computed at {mlcv_save_path}")
                cv = np.load(mlcv_save_path)
                print(f"> CV shape: {cv.shape}")
                print(f"> CV range: [{cv.min():.6f}, {cv.max():.6f}]")
                print(f"> CV mean: {cv.mean():.6f}, std: {cv.std():.6f}")
                
                plt.figure(figsize=(5, 4))
                plt.hist(cv, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel("CV")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                save_path_normal = f"./temp/cv_histogram_{method}_{molecule}.png"
                plt.savefig(save_path_normal, dpi=300, bbox_inches="tight")
                plt.close()
                
                plt.figure(figsize=(5, 4))
                plt.hist(cv, bins=50, alpha=0.7, color='skyblue', edgecolor='black', log=True)
                plt.xlabel("CV")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                save_path_log = f"./temp/cv_histogram_{method}_{molecule}_log.png"
                plt.savefig(save_path_log, dpi=300, bbox_inches="tight")
                plt.close()
                                
                wandb.init(
                    project="cv-analysis",
                    name=f"{method}-{molecule}",
                )
                wandb.log({
                    "molecule": molecule,
                    "method": method,
                    "cv/histogram": wandb.Image(save_path_normal),
                    "cv/histogram(log)": wandb.Image(save_path_log),
                    "cv/mean": cv.mean(),
                    "cv/std": cv.std(),
                })
                wandb.finish()
            
            else:
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