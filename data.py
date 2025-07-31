import torch
from torch.utils.data import Dataset

class TimelagDataset(Dataset):
    def __init__(
        self,
        cfg_data,
        device,
    ):
        self.cfg = cfg_data
        self.sequence = cfg_data.sequence
        self.representation = cfg_data.representation
        self.time_lag = cfg_data.time_lag
        self.dataset_size = cfg_data.dataset_size
        self.system_id = cfg_data.system_id
        self.data_dir = cfg_data.data_dir
        self.device = device
        
        self._load_data()
        
    def _load_data(self):
        if self.time_lag ==0:
            self.current_cad_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/current-cad.pt"
            self.current_pos_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/current-pos.pt"
            self.timelag_cad_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/current-cad.pt"
            self.timelag_pos_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/current-pos.pt"
        else:
            self.current_cad_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/current-cad.pt"
            self.current_pos_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/current-pos.pt"
            self.timelag_cad_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/lag{self.time_lag}-cad.pt"
            self.timelag_pos_path = f"{self.data_dir}/{self.system_id}-{self.dataset_size}/lag{self.time_lag}-pos.pt"
        
        self.current_cad = torch.load(self.current_cad_path, map_location=self.device)
        self.current_pos = torch.load(self.current_pos_path, map_location=self.device)
        self.timelag_cad = torch.load(self.timelag_cad_path, map_location=self.device)
        self.timelag_pos = torch.load(self.timelag_pos_path, map_location=self.device)

    def __len__(self):
        if self.representation == "cad":
            return len(self.current_cad)
        elif self.representation == "pos":
            return len(self.current_pos)
        elif self.representation == "cad-pos":
            return len(self.current_cad)
        else:
            raise ValueError(f"Invalid representation: {self.representation}")

    def __getitem__(self, idx):
        if self.representation == "cad":
            return {
                "current_data": self.current_cad[idx],
                "timelagged_data": self.timelag_cad[idx]
            }
        elif self.representation == "pos":
            return {
                "current_data": self.current_pos[idx],
                "timelagged_data": self.timelag_pos[idx]
            }
        elif self.representation == "cad-pos":
            return {
                "current_data": self.current_cad[idx],
                "timelagged_data": self.timelag_pos[idx],
            }
        else:
            raise ValueError(f"Invalid representation: {self.representation}")