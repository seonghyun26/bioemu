import os
import wandb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from .constant import *


def plot_phi_distribution(args, base_dir):
    wandb_image_list = []
    
    for seed in tqdm(
        range(0, args.seed + 1),
        desc="Plotting phi distribution",
        total=args.seed + 1
    ):
        # Load COLVAR data
        cv_dir = os.path.join(base_dir, 'log', args.date, str(seed), 'COLVAR')
        data = np.loadtxt(cv_dir, comments='#')
        
        # Read data
        with open(cv_dir, 'r') as file:
            first_line = file.readline().strip()
            keys = first_line.split()[2:]
        time_horizon = int(data[:, keys.index('time')][-1])
        phi = data[:, keys.index('phi')]
        cv = data[:,keys.index(cv_name[args.method])]
        time_ns = np.arange(0, time_horizon + 1) * 0.001
        
        # Plotting
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(cv.min(), cv.max())
        colors = cmap(norm(cv))
        
        # Plot data        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(time_ns, phi, c=colors, s=10)
        if args.method == "ref":
            scatter.set_clim(-np.pi, np.pi)
            cbar = plt.colorbar(
                scatter,
                ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            )
            cbar.ax.tick_params(labelsize=FONTSIZE_SMALL)
            cbar.set_label(bar_labels[args.method], fontweight="medium", fontsize=FONTSIZE)
            cbar.set_ticklabels(
                [r"$-\pi$", r"$-\pi$"+"/2", r"$0$", r"$\pi$"+"/2", r"$\pi$"]
            )
        # elif args.method == "DeepLDA":
        #     abs_max = np.max(np.abs(cv))
        #     scatter.set_clim(-abs_max, abs_max)
        #     cbar = plt.colorbar(
        #         scatter,
        #         ticks = [-1.0, -0.5, 0, 0.5, 1.0]
        #     )
        #     cbar.ax.tick_params(labelsize=FONTSIZE_SMALL)
        #     cbar.set_label(bar_labels[args.method], fontweight="medium", fontsize=FONTSIZE)
        # if args.method == "debug":
        #     fig, ax = plt.subplots(figsize=(2, 10))
        #     cmap = plt.get_cmap("viridis")
        #     norm = plt.Normalize(vmin=cv.min(), vmax=cv.max())
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     cbar = plt.colorbar(
        #         sm,
        #         cax=ax,
        #         orientation='vertical',
        #         aspect=200,              # Higher → thinner (default is ~20)
        #         shrink=0.3,             # 1.0 = full size; <1.0 = shorter
        #         ticks=[-1.0, -0.5, 0, 0.5, 1.0]
        #     )
        #     cbar.set_label('MLCV', fontsize=FONTSIZE)
        #     cbar.ax.tick_params(labelsize=FONTSIZE)
        #     plt.tight_layout()
            
        #     plot_path = os.path.join("./fig", 'phi-cv-colorbar2.png')
        #     plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        #     plt.savefig(plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', transparent=True)
        #     plt.close()
        #     print(f">> CV colorbar saved at {plot_path}")
        
        # Count phi
        phi_negative = phi[phi < 0].shape[0] / phi.shape[0]
        phi_positive = phi[phi > 0].shape[0] / phi.shape[0]
        print(f"{phi.shape[0]} points: {phi_positive * 100:.1f}% positive, {phi_negative * 100:.1f}% negative")
        
        # Customize plot
        plt.xlabel("Time (ns)", fontsize=FONTSIZE, fontweight="medium")
        plt.ylabel(r"$\phi$" + " (rad)", fontsize=FONTSIZE, fontweight="medium")
        plt.xlim(0,20)
        plt.ylim(-np.pi,np.pi)
        plt.xticks(np.arange(0, 21, 5),fontsize=FONTSIZE_SMALL) 
        plt.yticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            [r"$-\pi$", r"$-\pi$"+"/2", r"$0$", r"$\pi$"+"/2", r"$\pi$"],
            fontsize=FONTSIZE_SMALL
        )
        
        plt.tight_layout()
        plt.savefig(f'./fig/phi_{args.method}_{args.ns}_{seed}.png', dpi=300, bbox_inches="tight")
        plt.savefig(f'./fig/phi_{args.method}_{args.ns}_{seed}.pdf', dpi=300, bbox_inches="tight")
        wandb.log({
            f"phi_distribution/{seed}": wandb.Image(f"./fig/phi_{args.method}_{args.ns}_{seed}.png"),
            f"phi_positive/{seed}": phi_positive,
            f"phi_negative/{seed}": phi_negative,
        })
        print(f'Figure saved at ./fig/phi_{args.method}_{args.ns}_{seed}.png')
        plt.close()
        
    return wandb_image_list