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

from src.models.gru_baseline import GRUBaseline
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
    Set random seeds for reproducible training.

    Parameters
    ----------
    seed : int
        Seed value used for Python, NumPy, and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    """
    Load a processed dataset split from CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a processed CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with charttime converted to datetime.
    """

    df = pd.read_csv(path)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def save_scatter_plot(preds, targets, window_size, prediction_horizon):
    """
    Save a validation scatter plot of predicted versus true values.

    Parameters
    ----------
    preds : torch.Tensor
        Model predictions.

    targets : torch.Tensor
        True target values.

    window_size : int
        Input window size in hours.

    prediction_horizon : int
        Forecast horizon in hours.
    """

    import matplotlib.pyplot as plt

    preds_np = preds.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()

    plt.scatter(targets_np, preds_np, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"GRU Scatter: Window={window_size}, Horizon={prediction_horizon}")

    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir / f"gru_scatter_w{window_size}_h{prediction_horizon}.png")
    plt.close()


def save_checkpoint(model, window_size, prediction_horizon):
    """
    Save the best GRU model checkpoint for one experimental setting.

    Parameters
    ----------
    model : torch.nn.Module
        GRU model to save.

    window_size : int
        Input window size in hours.

    prediction_horizon : int
        Forecast horizon in hours.
    """

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"gru_w{window_size}_h{prediction_horizon}.pt"

    torch.save(model.state_dict(), checkpoint_path)

    print(f"Saved best GRU checkpoint to {checkpoint_path}")


def train_gru_for_setting(window_size, prediction_horizon):
    """
    Train and validate the GRU baseline for one configuration.

    Parameters
    ----------
    window_size : int
        Number of past hours used as input.

    prediction_horizon : int
        Number of hours ahead to predict.

    Returns
    -------
    list of dict
        Per-epoch training and validation metrics.
    """

    set_seed(SEED)

    train_df = load_dataset("data/processed/splits/train.csv")
    val_df = load_dataset("data/processed/splits/val.csv")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUBaseline(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = torch.nn.MSELoss()

    results = []

    best_val_loss = float("inf")
    best_preds = None
    best_targets = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_loss, preds, targets, last_inputs = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

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

        row = {
            "model": "GRU",
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

        row.update(epoch_per_variable_mae)
        results.append(row)

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

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best Val Loss={best_val_loss:.4f}"
            )
            break

    if best_preds is not None and best_targets is not None:
        save_scatter_plot(best_preds, best_targets, window_size, prediction_horizon)

    return results


def main():
    """
    Train the GRU baseline across all window sizes and prediction horizons.

    Saves all per-epoch results to outputs/metrics/gru_results.csv.
    """

    all_results = []

    for window_size in WINDOW_SIZES:
        for prediction_horizon in PREDICTION_HORIZONS:
            results = train_gru_for_setting(
                window_size=window_size,
                prediction_horizon=prediction_horizon,
            )
            all_results.extend(results)

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "gru_results.csv", index=False)

    print("\nSaved results to outputs/metrics/gru_results.csv")


if __name__ == "__main__":
    main()