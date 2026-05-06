import torch
import numpy as np


def mae(predictions, targets):
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions with shape (N, num_features).

    targets : torch.Tensor
        Ground-truth values with the same shape as predictions.

    Returns
    -------
    float
        Mean absolute error across all samples and features.
    """

    return torch.mean(torch.abs(predictions - targets)).item()


def rmse(predictions, targets):
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions with shape (N, num_features).

    targets : torch.Tensor
        Ground-truth values with the same shape as predictions.

    Returns
    -------
    float
        Root mean squared error across all samples and features.
    """

    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def pearson_corr(predictions, targets):
    """
    Compute Pearson correlation coefficient between predictions and targets.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions with shape (N, num_features).

    targets : torch.Tensor
        Ground-truth values with the same shape.

    Returns
    -------
    float
        Pearson correlation coefficient.

    Notes
    -----
    - Both tensors are flattened before computing correlation.
    - If either predictions or targets have zero variance,
      correlation is undefined and returns 0.0.
    """

    preds = predictions.detach().cpu().numpy().flatten()
    true = targets.detach().cpu().numpy().flatten()

    # Avoid division by zero in correlation computation
    if np.std(preds) == 0 or np.std(true) == 0:
        return 0.0

    return float(np.corrcoef(preds, true)[0, 1])


def per_variable_mae(predictions, targets, vital_columns):
    """
    Compute MAE separately for each vital sign.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions with shape (N, num_features).

    targets : torch.Tensor
        Ground-truth values with the same shape.

    vital_columns : list of str
        Names of the vital-sign features.

    Returns
    -------
    dict
        Dictionary mapping each vital sign to its MAE.
        Example: {"mae_heart_rate": 0.12, ...}
    """

    # Mean absolute error per feature dimension
    errors = torch.mean(torch.abs(predictions - targets), dim=0)

    return {
        f"mae_{vital}": errors[i].item()
        for i, vital in enumerate(vital_columns)
    }


def direction_accuracy(predictions, targets, last_inputs):
    """
    Compute directional accuracy of predictions.

    Direction is defined relative to the last observed input value.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions with shape (N, num_features).

    targets : torch.Tensor
        Ground-truth values with the same shape.

    last_inputs : torch.Tensor
        Last observed values from the input window, shape (N, num_features).

    Returns
    -------
    float
        Fraction of predictions that correctly match the direction of change.

    Notes
    -----
    Direction is computed as:
        sign(prediction - last_input)

    A prediction is considered correct if it matches the sign of:
        sign(target - last_input)

    Edge case:
    - If either difference is zero, sign = 0, which is treated as a valid match.
    """

    # Compute direction of change for predictions and targets
    pred_direction = torch.sign(predictions - last_inputs)
    true_direction = torch.sign(targets - last_inputs)

    # Compare predicted vs true direction
    correct = (pred_direction == true_direction).float()

    return correct.mean().item()