import numpy as np
import torch
import torch.nn as nn


class L1:
    """
    Mean absolute error (MAE)
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred-target))


class L2:
    """
    Mean square error (MSE)
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred-target)**2)

class RMSE:
    """
    Root mean square error
    """

    def __call__(self, pred:torch.Tensor, target: torch.Tensor)-> torch.Tensor:
        return torch.sqrt(torch.mean((pred - target)**2))

class debiased_l2:
    """
    Debiased MSE
    """

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, com_pred:torch.Tensor, com_target) -> torch.Tensor
        db_pred = torch.abs(pred - com_pred)
        db_target = torch.abs(target - com_target)
        return torch.mean(torch.abs(db_pred - db_target)**2) 