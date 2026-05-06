from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import SEED, VITAL_COLUMNS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
from src.data.window_dataset import VitalWindowDataset
from src.models.gru_baseline import GRUBaseline
from src.models.pc_gru import PredictiveCodingGRU
from src.training.train import train_one_epoch, evaluate
from src.evaluation.metrics import mae, rmse, pearson_corr, direction_accuracy



OUTPUT_PATH = Path("outputs/metrics/quick_run_results.csv")


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_vitals():
    """
    Create a small synthetic ICU-like dataset.

    This dataset mimics simple temporal trends with noise
    and is only used to verify that the pipeline runs.
    """

    rows = []

    for subject_id in [1, 2, 3, 4]:
        hadm_id = subject_id * 10
        stay_id = subject_id * 100

        for t in range(12):
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "stay_id": stay_id,
                    "charttime": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=t),
                    "heart_rate": 0.1 * t + np.random.normal(0, 0.05),
                    "systolic_bp": 1.0 - 0.03 * t + np.random.normal(0, 0.05),
                    "diastolic_bp": 0.5 - 0.02 * t + np.random.normal(0, 0.05),
                    "temperature": 0.02 * t + np.random.normal(0, 0.03),
                    "respiratory_rate": 0.08 * t + np.random.normal(0, 0.05),
                    "oxygen_saturation": 1.0 - 0.01 * t + np.random.normal(0, 0.02),
                }
            )

    return pd.DataFrame(rows)


def run_model(model_name, model, train_loader, val_loader, device):
    """
    Train and evaluate one model for the quick-run.
    """

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Single epoch (fast)
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

    val_loss, preds, targets, last_inputs = evaluate(
        model, val_loader, loss_fn, device
    )

    return {
        "model": model_name,
        "window_size_hours": 2,
        "prediction_horizon_hours": 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mae": mae(preds, targets),
        "rmse": rmse(preds, targets),
        "pearson": pearson_corr(preds, targets),
        "direction_accuracy": direction_accuracy(preds, targets, last_inputs),
    }


def main():
    set_seed()

    # Synthetic dataset (self-contained)
    df = make_synthetic_vitals()

    # Simple split
    train_df = df[df["subject_id"].isin([1, 2, 3])].copy()
    val_df = df[df["subject_id"].isin([4])].copy()

    window_size = 2
    prediction_horizon = 1
    batch_size = 4

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    # GRU
    gru_model = GRUBaseline(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )

    results.append(
        run_model("GRU-quick", gru_model, train_loader, val_loader, device)
    )

    # PC-GRU
    pc_gru_model = PredictiveCodingGRU(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=2,
    )

    results.append(
        run_model("PC-GRU-quick", pc_gru_model, train_loader, val_loader, device)
    )

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print("\nQuick run completed successfully.")
    print("\nResults:")
    for row in results:
        print(row)

    print(f"\nSaved quick-run results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
