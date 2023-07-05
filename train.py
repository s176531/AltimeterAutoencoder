from src.autoencoder import Encoder, Decoder
from src.data_setup import load_data
from src.criterion import Masked_Loss, GDL, Loss
from src.save_load_model import save_model

import torch
from torch import nn
from datetime import date
from pathlib import Path
import logging

def train_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASEPATH = Path(".")
    SAVEFOLDER = Path(".")
    SAVEFOLDER.mkdir(parents=True, exist_ok=True)

    use_masks = False
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
            train_loss = criterion(features, output, mask if use_masks else None)
            train_loss.backward()
            
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
                validation_loss += criterion(features, output, mask if use_masks else None).item()
        validation_loss /= len(validation_loader)

        logging.info(f"Epoch: {epoch} - Training loss: {training_loss} - Validation loss: {validation_loss}")
        print(f"Epoch: {epoch} - Training loss: {training_loss} - Validation loss: {validation_loss}")

        if epoch % save_epoch == 0:
            save_model(
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
                SAVEFOLDER / f"checkpoint_{epoch}.pkl"
            )

if __name__ == "__main__":
    train_model()