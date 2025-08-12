import os
import re
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .constant import *


def marginalize(free):    
    free = free.reshape(100, 100)
    free = -2.49 * np.logaddexp.reduce(-1 / 2.49 * free, 0)
    return free


def plot_fes_over_cv(args, base_dir):
    pmfs = []

    for seed in range(0, args.seed + 1):
        fes_dir = os.path.join(base_dir, 'log', args.date, str(seed), "fes")
        files = [
            f for f in os.listdir(fes_dir)
            if f.startswith("fes_") and f.endswith(".dat")
        ]
        pattern = re.compile(r"^fes_(\d+)\.dat$")
        last_fes_file = max(
            files,
            key=lambda f: int(pattern.match(f).group(1)) if pattern.match(f) else -1
        )
        
        fes_file = os.path.join(base_dir, 'log', args.date, str(seed), f"fes/{last_fes_file}")
        with open(fes_file, 'r') as file:
            first_line = file.readline().strip()
        keys = first_line.split()[2:]

        data = np.loadtxt(fes_file, comments='#')
        cv_idx = keys.index(cv_name[args.method])
        free_idx = keys.index('file.free')
        cv = data[:, cv_idx]
        free = data[:, free_idx]
        
        if args.method == 'ref':
            free = marginalize(free)
            cv = np.arange(-np.pi, np.pi, 2*np.pi/51)

        pmfs.append(free)
        
    pmfs = np.array(pmfs)
    mean_pmf = np.mean(pmfs, axis=0)
    mean_pmf = mean_pmf - mean_pmf.min()
    std_pmf = np.std(pmfs, axis=0)
    


    c = "#5684E9"
    plt.figure(figsize=(8, 5))
    plt.plot(cv, mean_pmf, color=c)
    plt.fill_between(cv, mean_pmf - std_pmf, mean_pmf + std_pmf, alpha=0.2, color=c)
    
    plt.xlabel(bar_labels[args.method], fontsize=FONTSIZE, fontweight="medium")
    plt.ylabel('FES (kJ/mol)', fontsize=FONTSIZE, fontweight="medium")
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(fontsize=FONTSIZE_SMALL)
    plt.yticks(fontsize=FONTSIZE_SMALL)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=2)
    plt.tight_layout()
    
    plt.savefig(f'./fig/fes_{args.method}_{args.ns}.png', dpi=300, bbox_inches="tight")
    plt.savefig(f'./fig/fes_{args.method}_{args.ns}.pdf', dpi=300, bbox_inches="tight")
    wandb.log({f"fes": wandb.Image(f'./fig/fes_{args.method}_{args.ns}.png')})
    print(f'Figure saved at ./fig/fes_{args.method}_{args.ns}.png')
    plt.show()
    plt.close()
    
    return
