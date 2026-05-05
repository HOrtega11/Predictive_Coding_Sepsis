
import torch
import numpy as np


def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets)).item()


def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def pearson_corr(predictions, targets):
    preds = predictions.detach().cpu().numpy().flatten()
    true = targets.detach().cpu().numpy().flatten()

    if np.std(preds) == 0 or np.std(true) == 0:
        return 0.0

    return float(np.corrcoef(preds, true)[0, 1])


def per_variable_mae(predictions, targets, vital_columns):
    errors = torch.mean(torch.abs(predictions - targets), dim=0)

    return {
        f"mae_{vital}": errors[i].item()
        for i, vital in enumerate(vital_columns)
    }


def direction_accuracy(predictions, targets, last_inputs):
    pred_direction = torch.sign(predictions - last_inputs)
    true_direction = torch.sign(targets - last_inputs)

    correct = (pred_direction == true_direction).float()

    return correct.mean().item()
