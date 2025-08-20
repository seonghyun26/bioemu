import os
import re
import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.interpolate import interp1d

from .constant import *

def load_fes_data(log_dir, method, idx):
    # Load file
    fes_file = os.path.join(log_dir, 'fes/fes_' + str(idx) + ".dat")
    fes_data = np.loadtxt(fes_file, comments='#')
    with open(fes_file, 'r') as file:
        first_line = file.readline().strip()

    # Load keys
    keys = first_line.split()[2:]
    phi_idx = keys.index(cv_name[method])
    free_idx = keys.index('file.free')
    
    # Loda data
    if method in ['phi','ref']:
        phi = fes_data[:, phi_idx]
        free = fes_data[:, free_idx]
        return phi, free
    
    else:
        # Load grid information
        cv_idx = keys.index(cv_name[method])
        cv_grid = fes_data[:, cv_idx]
        free_grid = fes_data[:, free_idx]
        
        # Load colvar data
        cv_file = os.path.join(log_dir, "COLVAR")
        with open(cv_file, 'r') as file:
            first_line = file.readline().strip()
            colvar_keys = first_line.split()[2:]
        colvar_cv_idx = colvar_keys.index(cv_name[method])
        colvar_phi_idx = colvar_keys.index('phi')
        cv_data = np.loadtxt(cv_file, comments='#')
        cv = cv_data[:, colvar_cv_idx]
        phi = cv_data[:, colvar_phi_idx]

        # Compute FES
        fes_interp = interp1d(
            cv_grid, 
            free_grid, 
            kind='linear', 
            fill_value="extrapolate"
        )
        free = fes_interp(cv)
        
        return phi, free


def calculate_delta_f(phi, free):    
    # Filter data based on conditions
    A = free[phi < 0]
    B = free[(phi > 0)]
    
    # Calculate free energies
    fesA = -2.49 * np.logaddexp.reduce(-1 / 2.49 * A)
    fesB = -2.49 * np.logaddexp.reduce(-1 / 2.49 * B)
    if np.isnan(fesB-fesA) or np.isinf(fesB-fesA):
        return np.nan
    return fesB - fesA


def plot_free_energy_difference(args, base_dir):
    # Load time information
    cv_dir = os.path.join(base_dir, 'log', args.date, '0', 'COLVAR')
    data = np.loadtxt(cv_dir, comments='#')
    with open(cv_dir, 'r') as file:
        first_line = file.readline().strip()
        keys = first_line.split()[2:]
    # time_horizon = int(data[:, keys.index('time')][-1]) // 200
    
    # Compute FES
    seed_delta_fs = []
    for seed in tqdm(
        range(0, args.seed + 1),
        desc = "Computing mean delta"
    ):
        log_dir = os.path.join(base_dir, 'log', args.date, str(seed))
        delta_fs = []
        
        # Compute delta F over time
        fes_dir = os.path.join(log_dir, "fes")
        files = [f for f in os.listdir(fes_dir) if os.path.isfile(os.path.join(fes_dir, f))]

        fes_num = files[-1]
        m = re.match(r"fes_(\d+)\.dat", fes_num)
        fes_num = int(m.group(1))
        for HILLS_idx in range(1, fes_num + 1):
            phi, free = load_fes_data(log_dir, args.method, HILLS_idx)
            fes = calculate_delta_f(phi, free)
            delta_fs.append(fes)
        seed_delta_fs.append(delta_fs)
    delta_fs = np.array(seed_delta_fs)
    mean_delta_fs = np.nanmean(delta_fs, axis=0)
    std_delta_fs = np.nanstd(delta_fs, axis=0)
    print(mean_delta_fs[-1], std_delta_fs[-1])
    
    cv_dir = os.path.join(base_dir, 'log', args.date, '0', 'COLVAR')
    with open(cv_dir, 'r') as file:
        first_line = file.readline().strip()
        keys = first_line.split()[2:]
    data = np.loadtxt(cv_dir, comments='#')
    time_horizon = int(data[:, keys.index('time')][-1]) / 1000 # unit in nano, since record in ps
    times = np.linspace(0, time_horizon, num=mean_delta_fs.shape[0])
    print(f"Times: {times}")
    
    c = "#5684E9"
    ref = 10.06
    mask = ~np.isnan(mean_delta_fs)
    mask = mask & (times > 3)
    yticks = [4, 8, 12, 16]
    
    plt.figure(figsize=(8, 5))
    if mask.sum() == 0:
        plt.plot(times, np.zeros_like(times), alpha=0.0)
    else:
        plt.plot(times[mask], mean_delta_fs[mask], color=c, linewidth=2)
        plt.fill_between(
            times[mask], 
            mean_delta_fs[mask] - std_delta_fs[mask], 
            mean_delta_fs[mask] + std_delta_fs[mask], 
            alpha=0.2,
            color=c,
            linewidth=2
        )
    plt.xlim(3, time_horizon)
    plt.ylim(yticks[0] - 1, yticks[-1] + 1)
    plt.axhline(y=ref, color='#C10035', linestyle='--', label='GT', linewidth=2)
    plt.fill_between(times, ref - 0.5, ref + 0.5, color='#C10035', alpha=0.2)
    plt.xlabel('Time (ns)', fontsize=FONTSIZE, fontweight="medium")
    plt.ylabel(r'$\Delta F$' + ' (kJ/mol)', fontsize=FONTSIZE, fontweight="medium")
    # plt.title(r'$\Delta F$' + f' over {args.method}', fontsize=20, fontweight="medium")
    plt.xticks(fontsize=FONTSIZE_SMALL)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.yticks(yticks, fontsize=FONTSIZE_SMALL)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'./fig/deltaf_{args.method}_{args.ns}.png', dpi=300, bbox_inches="tight")
    plt.savefig(f'./fig/deltaf_{args.method}_{args.ns}.pdf', dpi=300, bbox_inches="tight")
    wandb.log({
        f"delta_f": wandb.Image(f'./fig/deltaf_{args.method}_{args.ns}.png'),
        f"mean_delta_f": mean_delta_fs[-1],
        f"std_delta_f": std_delta_fs[-1]
    })
    print(f'Figure saved at ./fig/deltaf_{args.method}_{args.ns}.png')
    plt.show()
    plt.close()
    
    return