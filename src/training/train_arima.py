import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.arima_baseline import ARIMABaseline
from config import (
    VITAL_COLUMNS,
    ARIMA_ORDER,
    PREDICTION_HORIZONS,
    VAL_DATA_PATH,
    ARIMA_OUTPUT_PATH,
)
from src.evaluation.metrics import (
    mae,
    rmse,
    pearson_corr,
    per_variable_mae,
    direction_accuracy,
)


def load_dataset(path):
    df = pd.read_csv(path)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def run_arima(horizon):
    val_df = load_dataset(VAL_DATA_PATH)

    model = ARIMABaseline(order=ARIMA_ORDER)

    all_preds = []
    all_targets = []
    all_last_inputs = []

    for _, stay_df in val_df.groupby(["subject_id", "hadm_id", "stay_id"]):
        stay_df = stay_df.sort_values("charttime")

        preds, targets, last_inputs = model.predict_patient(
            stay_df,
            VITAL_COLUMNS,
            horizon,
        )

        if len(preds) == 0:
            continue

        all_preds.append(preds)
        all_targets.append(targets)
        all_last_inputs.append(last_inputs)

    if not all_preds:
        raise ValueError(f"No ARIMA predictions generated for horizon={horizon}")

    preds = torch.tensor(np.vstack(all_preds), dtype=torch.float32)
    targets = torch.tensor(np.vstack(all_targets), dtype=torch.float32)
    last_inputs = torch.tensor(np.vstack(all_last_inputs), dtype=torch.float32)

    results = {
        "model": "ARIMA",
        "prediction_horizon_hours": horizon,
        "mae": mae(preds, targets),
        "rmse": rmse(preds, targets),
        "pearson": pearson_corr(preds, targets),
        "direction_accuracy": direction_accuracy(preds, targets, last_inputs),
    }

    results.update(per_variable_mae(preds, targets, VITAL_COLUMNS))

    return results


def main():
    warnings.filterwarnings("ignore")

    all_results = []

    for horizon in PREDICTION_HORIZONS:
        results = run_arima(horizon)
        all_results.append(results)
        print(results)

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(ARIMA_OUTPUT_PATH, index=False)

    print(f"\nSaved results to {ARIMA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()