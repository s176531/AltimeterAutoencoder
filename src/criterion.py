import torch
from torch import nn
import warnings
from typing import Any, List, Type

class Loss():
    def __init__(self, losses: List[nn.modules.loss._Loss]):
        self.losses = losses

    def __call__(self, ground_truth, predictions, mask):
        return sum([loss(ground_truth, predictions, mask) for loss in self.losses])

class Masked_Loss(nn.Module):
    def __init__(self, loss: Type[nn.modules.loss._Loss]):
        super().__init__()
        self.loss = loss()
        
    def forward(self, ground_truth, predictions, mask = None):
        """
        Maskes the output and target based on the mask.
        The mask is True where the values should be ignored.
        """
        total_loss = torch.tensor(0, device = ground_truth.device, dtype = ground_truth.dtype)
        for day_idx in range(ground_truth.shape[1]):
            if mask is None:
                loss = self.loss(ground_truth[:, day_idx], predictions[:, day_idx])
            else:
                current_mask = ~mask[:, day_idx]
                loss = self.loss(ground_truth.squeeze(2)[:, day_idx][current_mask], predictions.squeeze(2)[:, day_idx][current_mask])
            total_loss += loss
        return total_loss / ground_truth.shape[1]

class GDL(nn.Module):
    def __init__(self, alpha = 1, print_warning = True):
        """
        Gradient Difference Loss
        Args:
            alpha: hyper parameter of GDL loss, float
        """
        super().__init__()
        self.alpha = alpha
        self.print_warning = print_warning

    def __call__(self, ground_truth, predictions, mask = None):
        """
        predictions --- tensor with shape (batch_size, frames, channels, height, width)
        ground_truth --- tensor with shape (batch_size, frames, channels, height, width)
        """

        ground_truth = ground_truth.flatten(0, 1)
        predictions = predictions.flatten(0, 1)
        mask = mask.flatten(0, 1)

        ground_truth_i1 = ground_truth[:, :, 1:, :]
        ground_truth_i2 = ground_truth[:, :, :-1, :]
        ground_truth_j1 = ground_truth[:, :, :, :-1]
        ground_truth_j2 = ground_truth[:, :, :, 1:]

        predictions_i1 = predictions[:, :, 1:, :]
        predictions_i2 = predictions[:, :, :-1, :]
        predictions_j1 = predictions[:, :, :, :-1]
        predictions_j2 = predictions[:, :, :, 1:]

        if mask is None:
            if self.print_warning:
                warnings.warn("Mask was not set in GDL")
        else:
            mask_i1 = mask[:, :, 1:, :]
            mask_i2 = mask[:, :, :-1, :]
            mask_j1 = mask[:, :, :, :-1]
            mask_j2 = mask[:, :, :, 1:]
            mask1 = ~(mask_i1 | mask_i2)
            mask2 = ~(mask_j1 | mask_j2)
            

        term1 = torch.abs(ground_truth_i1 - ground_truth_i2)
        term2 = torch.abs(predictions_i1 - predictions_i2)
        term3 = torch.abs(ground_truth_j1 - ground_truth_j2)
        term4 = torch.abs(predictions_j1 - predictions_j2)

        gdl1 = torch.pow(torch.abs(term1 - term2), self.alpha)
        gdl2 = torch.pow(torch.abs(term3 - term4), self.alpha)

        if mask is None:
            gdl1 = gdl1.mean()
            gdl2 = gdl2.mean()
        else:
            gdl1 = gdl1[mask1].mean()
            gdl2 = gdl2[mask2].mean()
        gdl_loss = gdl1 + gdl2
        
        return gdl_loss