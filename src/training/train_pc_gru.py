import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    SEED,
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    VITAL_COLUMNS,
    WINDOW_SIZES,
    PREDICTION_HORIZONS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    DROPOUT,
)

from src.models.pc_gru import PredictiveCodingGRU
from src.data.window_dataset import VitalWindowDataset
from src.training.train import train_one_epoch, evaluate
from src.evaluation.metrics import (
    mae,
    rmse,
    pearson_corr,
    per_variable_mae,
    direction_accuracy,
)


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed used for Python, NumPy, and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA seeds when a GPU is available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    """
    Load a processed split CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with charttime converted to datetime format.
    """

    df = pd.read_csv(path)

    # Ensure charttime is treated as a timestamp rather than plain text.
    df["charttime"] = pd.to_datetime(df["charttime"])

    return df


def save_scatter_plot(preds, targets, window_size, prediction_horizon):
    """
    Save a validation scatter plot of predicted versus true values.

    Parameters
    ----------
    preds : torch.Tensor
        Model predictions from the validation set.

    targets : torch.Tensor
        True target values from the validation set.

    window_size : int
        Input window size in hours.

    prediction_horizon : int
        Prediction horizon in hours.

    Notes
    -----
    The plot is saved to outputs/plots/.
    """

    import matplotlib.pyplot as plt

    # Convert tensors to flattened NumPy arrays for plotting.
    preds_np = preds.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()

    plt.scatter(targets_np, preds_np, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"PC-GRU Scatter: Window={window_size}, Horizon={prediction_horizon}")

    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir / f"pc_gru_scatter_w{window_size}_h{prediction_horizon}.png")
    plt.close()


def save_checkpoint(model, window_size, prediction_horizon):
    """
    Save the current PC-GRU model weights.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PC-GRU model.

    window_size : int
        Input window size in hours.

    prediction_horizon : int
        Prediction horizon in hours.

    Notes
    -----
    A separate checkpoint is saved for each window-size and horizon setting.
    """

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"pc_gru_w{window_size}_h{prediction_horizon}.pt"

    torch.save(model.state_dict(), checkpoint_path)

    print(f"Saved best PC-GRU checkpoint to {checkpoint_path}")


def train_pc_gru_for_setting(window_size, prediction_horizon):
    """
    Train and evaluate PC-GRU for one window-size / horizon configuration.

    Parameters
    ----------
    window_size : int
        Length of the input sequence in hours.

    prediction_horizon : int
        Number of hours ahead to predict.

    Returns
    -------
    list of dict
        Per-epoch training and validation results. Each dictionary contains
        loss values and evaluation metrics for one epoch.

    Process
    -------
    1. Load train and validation splits.
    2. Build windowed datasets.
    3. Initialize the PC-GRU model.
    4. Train for multiple epochs.
    5. Evaluate on validation data after each epoch.
    6. Save the best checkpoint based on validation loss.
    7. Stop early if validation loss stops improving.
    """

    set_seed(SEED)

    # Load preprocessed patient-level train/validation splits.
    train_df = load_dataset("data/processed/splits/train.csv")
    val_df = load_dataset("data/processed/splits/val.csv")

    # Construct fixed-length input windows and horizon-specific targets.
    train_dataset = VitalWindowDataset(
        dataframe=train_df,
        vital_columns=VITAL_COLUMNS,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
    )

    val_dataset = VitalWindowDataset(
        dataframe=val_df,
        vital_columns=VITAL_COLUMNS,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
    )

    # Dataloaders batch the windowed examples for training and validation.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize predictive-coding-inspired GRU model.
    model = PredictiveCodingGRU(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=2,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Forecasting is trained using mean squared error.
    loss_fn = torch.nn.MSELoss()

    results = []

    # Track best validation loss for checkpointing and early stopping.
    best_val_loss = float("inf")
    best_preds = None
    best_targets = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        # Train model for one full pass over the training set.
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        # Evaluate model on validation set.
        val_loss, preds, targets, last_inputs = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        # Compute validation metrics.
        epoch_mae = mae(preds, targets)
        epoch_rmse = rmse(preds, targets)
        epoch_pearson = pearson_corr(preds, targets)
        epoch_direction_acc = direction_accuracy(preds, targets, last_inputs)
        epoch_per_variable_mae = per_variable_mae(preds, targets, VITAL_COLUMNS)

        print(
            f"Window={window_size}h | Horizon={prediction_horizon}h | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"MAE={epoch_mae:.4f} | "
            f"RMSE={epoch_rmse:.4f} | "
            f"Pearson={epoch_pearson:.4f} | "
            f"Direction Acc={epoch_direction_acc:.4f}"
        )

        # Store epoch-level results for later tables and plots.
        row = {
            "model": "PC-GRU",
            "window_size_hours": window_size,
            "prediction_horizon_hours": prediction_horizon,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mae": epoch_mae,
            "rmse": epoch_rmse,
            "pearson": epoch_pearson,
            "direction_accuracy": epoch_direction_acc,
        }

        # Add per-vital-sign MAE metrics.
        row.update(epoch_per_variable_mae)
        results.append(row)

        # Save checkpoint whenever validation loss improves enough.
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_preds = preds
            best_targets = targets
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                window_size=window_size,
                prediction_horizon=prediction_horizon,
            )
        else:
            epochs_without_improvement += 1

        # Stop training if validation loss fails to improve for too long.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best Val Loss={best_val_loss:.4f}"
            )
            break

    # Save scatter plot using predictions from the best validation-loss epoch.
    if best_preds is not None and best_targets is not None:
        save_scatter_plot(best_preds, best_targets, window_size, prediction_horizon)

    return results


def main():
    """
    Train PC-GRU across all configured window sizes and prediction horizons.

    Results from all experimental settings are saved to:

    outputs/metrics/pc_gru_results.csv
    """

    all_results = []

    # Run full grid of window-size and prediction-horizon combinations.
    for window_size in WINDOW_SIZES:
        for prediction_horizon in PREDICTION_HORIZONS:
            results = train_pc_gru_for_setting(
                window_size=window_size,
                prediction_horizon=prediction_horizon,
            )
            all_results.extend(results)

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all per-epoch results for later aggregation and plotting.
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "pc_gru_results.csv", index=False)

    print("\nSaved results to outputs/metrics/pc_gru_results.csv")


if __name__ == "__main__":
    main()