import torch
from pathlib import Path
from .autoencoder import Encoder, Decoder
from datetime import date, datetime
from typing import Tuple

def save_model(
        encoder: Encoder,
        decoder: Decoder,
        epoch: int,
        training_loss: float,
        validation_loss: float,
        train_start: date,
        train_end: date,
        validation_end: date,
        fill_nan: float,
        input_channels: int,
        feature_dimension: int,
        learning_rate: float,
        optimizer: torch.optim.Optimizer,
        save_path: Path
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

def load_model(path: Path, device: torch.device) -> Tuple[Encoder, Decoder, torch.optim.Optimizer, float, float, float, int, date, date, date, float, float]:
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
    min_value = checkpoint['min']
    difference = checkpoint['difference']
    
    # Dataset parameters
    return encoder, decoder, optimizer, training_loss, validation_loss, fill_nan, epoch, train_start, train_end, validation_end, min_value, difference
