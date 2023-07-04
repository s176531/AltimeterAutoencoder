from src.autoencoder import Encoder, Decoder
from src.data_setup import SLADataset
from src.regressor import MetaRegression, Regression
from src.criterion import Masked_Loss, GDL, Loss

import torch
from torch import nn
from torch.utils.data import DataLoader

import xarray as xr
import numpy as np
from datetime import date, datetime
from pathlib import Path
from typing import Type, Tuple
import logging

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
            regressor = MetaRegression.load(file)
    else:
        regressor = fit_regressor(times[bool_train], sla[bool_train], save_path)
    
    sla -= regressor.predict(times).reshape(*sla.shape)

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
    

def fit_regressor(times, sla, save_path: Path):
    # Create and fit model
    function_kwargs = {"fit_type": ("poly", "fourier"), "period": 1, "deg": (1,1)}
    regressor = MetaRegression(Regression, function_kwargs, 0)
    regressor.fit(times, sla.reshape(sla.shape[0], -1))

    # Save metaregressor
    with open(save_path, 'wb') as file:
        regressor.save(file)
    
    return regressor

def save_model(
        encoder,
        decoder,
        epoch,
        training_loss,
        validation_loss,
        train_start,
        train_end,
        validation_end,
        fill_nan,
        input_channels,
        feature_dimension,
        learning_rate,
        optimizer,
        save_path
    ):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "validation_end": validation_end.strftime("%Y-%m-%d"),
            "fill_nan": fill_nan,
            "input_channels": input_channels,
            "feature_dimension": feature_dimension,
            "learning_rate": learning_rate,
            "optimizer": optimizer.state_dict(),
        },
        save_path
    )

def load_model(path: Path, device: torch.device) -> Tuple[nn.Module,nn.Module,torch.optim.Optimizer,Loss,Loss,int,int,date,date,date]:
    """Loads the model, optimizer and loss from the path"""
    checkpoint = torch.load(path, map_location=device)
    
    # Encoder
    encoder = Encoder(checkpoint['input_channels'], checkpoint['feature_dimension'])
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to(device)
    
    # Decoder
    decoder = Decoder(checkpoint['input_channels'], checkpoint['feature_dimension'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(params = list(encoder.parameters()) + list(decoder.parameters()), lr=checkpoint["learning_rate"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Loss
    training_loss = checkpoint["training_loss"]
    validation_loss = checkpoint["validation_loss"]

    # Misc
    fill_nan = checkpoint["fill_nan"]
    epoch = checkpoint["epoch"]
    train_start = datetime.strptime(checkpoint["train_start"], "%Y-%m-%d").date()
    train_end = datetime.strptime(checkpoint["train_end"], "%Y-%m-%d").date()
    validation_end = datetime.strptime(checkpoint["validation_end"], "%Y-%m-%d").date()
    
    # Dataset parameters
    return encoder, decoder, optimizer, training_loss, validation_loss, fill_nan, epoch, train_start, train_end, validation_end

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASEPATH = Path(".")
    SAVEFOLDER = Path(".")
    SAVEFOLDER.mkdir(parents=True, exist_ok=True)

    epochs = 20
    fill_nan = 0
    batch_size = 5
    save_epoch = 10

    train_start = date(2006, 1, 1)
    train_end = date(2014, 1, 1)
    validation_end = date(2019, 1, 1)

    data_path = BASEPATH / "without_polar_v6_mss21.nc"
    save_path = SAVEFOLDER / "Regression.pkl"

    train_loader, validation_loader, test_loader, lat, lon = load_data(
        train_start, 
        train_end, 
        validation_end,
        data_path,
        save_path,
        batch_size,
        fill_nan
    )

    logging.basicConfig(
        level=logging.INFO,
        datefmt='%a, %d %b %Y %H:%M:%S',
        format='%(asctime)s - %(message)s',
        filename=(SAVEFOLDER / 'train_log.log').as_posix(),
        filemode='a'
    )

    input_channels = 1
    feature_dimension = 128
    learning_rate = 1e-5

    encoder = Encoder(input_channels=input_channels, feature_dimension=feature_dimension).to(DEVICE)
    decoder = Decoder(output_channels=input_channels, feature_dimension=feature_dimension).to(DEVICE)

    optimizer = torch.optim.Adam(params = list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    mse_loss = Masked_Loss(nn.MSELoss)
    gdl_loss = GDL()
    criterion = Loss([mse_loss, gdl_loss])

    encoder.train()
    decoder.train()

    for epoch in range(1,epochs+1):
        training_loss = 0
        validation_loss = 0

        for features, mask, _ in train_loader:
            features = features.to(DEVICE)
            mask = mask.to(DEVICE)

            # encode-decode training data
            feature_space = encoder(features)
            output = decoder(feature_space)

            # evaluate loss
            train_loss = criterion(features, output, mask)
            train_loss.backwards()
            
            # update optimizer
            optimizer.step()
            optimizer.zero_grad()

            training_loss += train_loss.item()
        training_loss /= len(train_loader)

        with torch.no_grad():
            encoder.eval()
            decoder.eval()

            for features, mask, _ in validation_loader:
                features = features.to(DEVICE)
                mask = mask.to(DEVICE)

                # encode-decode training data
                feature_space = encoder(features)
                output = decoder(feature_space)

                # evaluate loss
                validation_loss += criterion(features, output, mask).item()
        validation_loss /= len(validation_loader)

        logging.info(f"Epoch: {epoch} - Training loss: {training_loss} - Validation loss: {validation_loss}")
        print(f"Epoch: {epoch} - Training loss: {training_loss} - Validation loss: {validation_loss}")

        if epoch % save_epoch == 0:
            save_model(encoder,
                decoder,
                epoch,
                training_loss,
                validation_loss,
                train_start,
                train_end,
                validation_end,
                fill_nan,
                input_channels,
                feature_dimension,
                learning_rate,
                optimizer,
                SAVEFOLDER / f"checkpoint_{epoch}.pkl"
            )

if __name__ == "__main__":
    main()