import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import date
from pathlib import Path
import xarray as xr
from . import _types, regressor


class SLADataset(Dataset):
    """Constructs the SLA dataset"""
    def __init__(self, x: _types.float_like, t: _types.int_like, fill_nan: float):
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
        mask = self.mask[idx].unsqueeze(0)
        return features.unsqueeze(0), mask, features_time

def load_data(
        train_start: date, 
        train_end: date, 
        validation_end: date,
        data_path: Path,
        save_path: Path,
        batch_size: int,
        fill_nan: float = 0,
        is_trained: bool = False
    ):

    with xr.open_dataset(data_path, engine="netcdf4") as file:
        file = file.sortby('time')
        sla = file['sla21'].data[:, :-1]
        times = file['time'].data
        lat = file['Latitude'].data[:-1]
        lon = file['Longitude'].data[:-1]

    # Set train, validation and test intervals
    train_start_np = np.array(train_start).astype("datetime64[ns]")
    train_end_np = np.array(train_end).astype("datetime64[ns]")
    validation_end_np = np.array(validation_end).astype("datetime64[ns]")

    # Save times
    bool_train = (times > train_start_np) & (times <= train_end_np)
    bool_validation = (times > train_end_np) & (times <= validation_end_np)
    bool_test = times > validation_end_np

    if is_trained:
        with open(save_path, 'rb') as file:
            regression = regressor.MetaRegression.load(file)
    else:
        regression = regressor.fit_regressor(times[bool_train], sla[bool_train], save_path)
    
    sla -= regression.predict(times).reshape(*sla.shape)

    # Bool mask time
    train_time = times[bool_train].astype("datetime64[D]").astype(int)
    validation_time = times[bool_validation].astype("datetime64[D]").astype(int)
    test_time = times[bool_test].astype("datetime64[D]").astype(int)

    # Bool mask sla
    train_features = sla[bool_train]
    validation_features = sla[bool_validation]
    test_features = sla[bool_test]

    # Kwargs to dataloaders
    kwargs_dataloader = {
        'shuffle': False,
        'batch_size': batch_size
    }

    # Dataloders
    train_loader = DataLoader(SLADataset(train_features, train_time, fill_nan), **kwargs_dataloader)
    validation_loader = DataLoader(SLADataset(validation_features, validation_time, fill_nan), **kwargs_dataloader)
    test_loader = DataLoader(SLADataset(test_features, test_time, fill_nan), **kwargs_dataloader)

    return train_loader, validation_loader, test_loader, lat, lon