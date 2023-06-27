import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import date
from torch.utils.data import Dataset


class SLADataset(Dataset):
    """Constructs the SLA dataset"""
    def __init__(self, x, t, fill_nan):
        # Convert to float32 (float64 is overkill)
        x_32 = x.astype(np.float32)
        
        self.mask = torch.tensor(np.isnan(x_32), dtype=torch.bool)
        self.x = torch.nan_to_num(torch.tensor(x_32, dtype=torch.float32), nan = fill_nan)
        self.t = torch.tensor(t)
        self.fill_nan = fill_nan
        self._len = len(self.x)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Features
        features = self.x[idx]
        features_time = self.t[idx]
        
        # Target
        mask = self.mask[idx]
        return features.unsqueeze(0), mask, features_time

