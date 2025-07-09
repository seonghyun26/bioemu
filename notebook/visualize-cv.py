import numpy as np
import mdtraj as md
import nglview as nv
import matplotlib.pyplot as plt

import pyemma
import pickle
import torch
import lightning
import torch
from mlcolvar.core.transform import Transform


from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform import Statistics
from matplotlib.colors import LogNorm
from tqdm import tqdm


np.bool = np.bool_
MLCV_DIM = 2
date = "0707_1532"



def sanitize_range(range: torch.Tensor):
    """Sanitize

    Parameters
    ----------
    range : torch.Tensor
        range to be used for standardization

    """

    if (range < 1e-6).nonzero().sum() > 0:
        print(
            "[Warning] Normalization: the following features have a range of values < 1e-6:",
            (range < 1e-6).nonzero(),
        )
    range[range < 1e-6] = 1.0

    return range


class PostProcess(Transform):
    def __init__(
        self,
        stats = None,
        reference_frame_cv = None,
        feature_dim = 1,
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("range", torch.ones(feature_dim))
        
        if stats is not None:
            min = stats["min"]
            max = stats["max"]
            self.mean = (max + min) / 2.0
            range = (max - min) / 2.0
            self.range = sanitize_range(range)
        
        if reference_frame_cv is not None:
            self.register_buffer(
                "flip_sign",
                torch.ones(1) * -1 if reference_frame_cv < 0 else torch.ones(1)
            )
        else:
            self.register_buffer("flip_sign", torch.ones(1))
        
    def forward(self, x):
        x = x.sub(self.mean).div(self.range)
        x = x * self.flip_sign
        
        return x


class MLCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]

    def __init__(
        self,
        encoder_layers: list,
        options: dict = None,
        **kwargs,
    ):
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)
        # ======= OPTIONS =======
        options = self.parse_options(options)
        
        # ======= BLOCKS =======
        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])
        
        
save_path  = f"/home/shpark/prj-mlcv/lib/bioemu/model/{date}/mlcv.pt"
model_state = torch.load(save_path)
mlcv_state_dict = model_state["mlcv_state_dict"]
encoder_layers = [45, 30, 30, MLCV_DIM]
options = {
	"encoder": {
		"activation": "tanh",
		"dropout": [0.1, 0.1, 0.1]
	},
	"norm_in": {
	},
}
mlcv_model = MLCV(
	encoder_layers = encoder_layers,
	options = options
)
mlcv_model.load_state_dict(mlcv_state_dict)
print(mlcv_model)

molecule = "CLN025"
simulation_idx = 0
simulation_num = 0
cln025_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-{simulation_idx}-protein/{molecule}-{simulation_idx}-CAdistance-switch.pt"
ca_distance_data = torch.load(cln025_cad_path)

# Alpha carbon distances
pdb_path = "/home/shpark/prj-mlcv/lib/DESRES/data/CLN025.pdb"
state_traj = md.load(pdb_path)
ca_atoms = state_traj.topology.select("name CA")
n_atoms = len(ca_atoms)	
atom_pairs = []
for i in range(n_atoms):
    for j in range(i+1, n_atoms):
        atom_pairs.append([ca_atoms[i], ca_atoms[j]])
print(atom_pairs)

# Load traj data
for simulation_idx in range(simulation_num + 1):
	traj_list = []
	for i in tqdm(
		range(53),
		desc="Loading trajectories"
	):
		file_idx = f"{i:03d}"
		traj = md.load_dcd(
			f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_CLN025-{simulation_idx}-protein/CLN025-{simulation_idx}-protein/{molecule}-{simulation_idx}-protein-{file_idx}.dcd",
			top=pdb_path
		)
		traj_list.append(traj)
	all_traj = md.join(traj_list)

	feat_dist = pyemma.coordinates.featurizer(pdb_path)
	feat_dist.add_distances(indices=atom_pairs)
	feature_distances = feat_dist.transform(all_traj)
	feature_switch_distances = (1 - np.power(feature_distances / 0.8, 6)) / (1 - np.power(feature_distances / 0.8, 12))
	print(feature_switch_distances.shape)
 
with open(f'/home/shpark/prj-mlcv/lib/DESRES/data/CLN025_tica_model_switch.pkl', 'rb') as f:
    tica = pickle.load(f)
print(tica)


mlcv_model.eval()
cv = mlcv_model(torch.from_numpy(feature_switch_distances))
cv = cv.detach().cpu().numpy()
stats = Statistics(torch.from_numpy(cv).cpu()).to_dict()
mlcv_model.postprocessing = PostProcess(stats).to(mlcv_model.device)
postprocessed_cv = mlcv_model(torch.from_numpy(feature_switch_distances))
print(postprocessed_cv.max())
print(postprocessed_cv.min())

tica_data = tica.transform(feature_switch_distances)
x = tica_data[:, 0]
y = tica_data[:, 1]
print(tica_data.shape)

# State information
# state_traj = md.load(pdb_path)
# state_feat = feat_dist.transform(state_traj)
# state_feat_switch = (1 - np.power(state_feat / 0.8, 6)) / (1 - np.power(state_feat / 0.8, 12))
# tica_state = tica_obj.transform(state_feat_switch)

# Plot
# postprocessed_cv *= -1
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
hb = ax.hexbin(
	x, y, C=postprocessed_cv[:, 0].detach().cpu().numpy(),  # data
	gridsize=200,                     # controls resolution
	reduce_C_function=np.mean,       # compute average per hexagon
	cmap='viridis',                  # colormap
)
plt.colorbar(hb)
plt.xlabel("TIC 1")
plt.ylabel("TIC 2")
# plt.gca().invert_yaxis()
plt.show()

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
hb = ax.hexbin(
	x, y, C=postprocessed_cv[:, 1].detach().cpu().numpy(),  # data
	gridsize=200,                     # controls resolution
	reduce_C_function=np.mean,       # compute average per hexagon
	cmap='viridis',                  # colormap
)
plt.colorbar(hb)
plt.xlabel("TIC 1")
plt.ylabel("TIC 2")
# plt.gca().invert_yaxis()
plt.show()
plt.savefig(f"./test.png")