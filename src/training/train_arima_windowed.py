
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    VITAL_COLUMNS,
    ARIMA_ORDER,
    WINDOW_SIZES,
    PREDICTION_HORIZONS,
    VAL_DATA_PATH,
)

from src.models.arima_baseline import ARIMABaseline
from src.evaluation.metrics import (
    mae,
    rmse,
    pearson_corr,
    per_variable_mae,
    direction_accuracy,
)


OUTPUT_PATH = "outputs/metrics/arima_windowed_results.csv"


def load_dataset(path):
    df = pd.read_csv(path)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def predict_patient_windowed(model, patient_df, vital_columns, window_size, horizon):
    preds = []
    targets = []
    last_inputs = []

    patient_df = patient_df.sort_values("charttime")
    data = patient_df[vital_columns].values

    for i in range(window_size - 1, len(data) - horizon):
        history = data[i - window_size + 1 : i + 1]

        pred_step = []

        for v in range(len(vital_columns)):
            series = history[:, v]
            forecast = model.forecast_series(series, steps=horizon)
            pred_step.append(forecast[-1])

        preds.append(pred_step)
        targets.append(data[i + horizon])
        last_inputs.append(data[i])

    return (
        np.array(preds),
        np.array(targets),
        np.array(last_inputs),
    )


def run_arima_windowed(window_size, horizon):
    val_df = load_dataset(VAL_DATA_PATH)

    model = ARIMABaseline(order=ARIMA_ORDER)

    all_preds = []
    all_targets = []
    all_last_inputs = []

    for _, stay_df in val_df.groupby(["subject_id", "hadm_id", "stay_id"]):
        preds, targets, last_inputs = predict_patient_windowed(
            model=model,
            patient_df=stay_df,
            vital_columns=VITAL_COLUMNS,
            window_size=window_size,
            horizon=horizon,
        )

        if len(preds) == 0:
            continue

        all_preds.append(preds)
        all_targets.append(targets)
        all_last_inputs.append(last_inputs)

    if not all_preds:
        raise ValueError(
            f"No ARIMA-windowed predictions generated for "
            f"window={window_size}, horizon={horizon}"
        )

    preds = torch.tensor(np.vstack(all_preds), dtype=torch.float32)
    targets = torch.tensor(np.vstack(all_targets), dtype=torch.float32)
    last_inputs = torch.tensor(np.vstack(all_last_inputs), dtype=torch.float32)

    results = {
        "model": "ARIMA-windowed",
        "window_size_hours": window_size,
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

    for window_size in WINDOW_SIZES:
        for horizon in PREDICTION_HORIZONS:
            print(
                f"Running ARIMA-windowed | "
                f"Window={window_size}h | Horizon={horizon}h"
            )

            results = run_arima_windowed(
                window_size=window_size,
                horizon=horizon,
            )

            all_results.append(results)
            print(results)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
