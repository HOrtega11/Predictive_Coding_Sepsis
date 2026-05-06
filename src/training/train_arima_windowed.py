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
    """
    Load a processed dataset split from CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the processed CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with charttime converted to datetime format.
    """

    df = pd.read_csv(path)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def predict_patient_windowed(model, patient_df, vital_columns, window_size, horizon):
    """
    Generate fixed-window ARIMA predictions for one ICU stay.

    Parameters
    ----------
    model : ARIMABaseline
        ARIMA baseline object used to forecast each vital sign independently.

    patient_df : pandas.DataFrame
        DataFrame containing one ICU stay's time-ordered vital-sign records.

    vital_columns : list of str
        Names of the vital-sign columns to forecast.

    window_size : int
        Number of past time steps used as ARIMA history.

    horizon : int
        Number of future time steps to forecast.

    Returns
    -------
    tuple of np.ndarray
        preds : np.ndarray
            Predicted vital signs with shape
            (num_windows, num_vitals).

        targets : np.ndarray
            True future vital signs with shape
            (num_windows, num_vitals).

        last_inputs : np.ndarray
            Last observed input values before prediction, used for
            directional accuracy.

    Notes
    -----
    For each prediction point i:
    - history = data[i - window_size + 1 : i + 1]
    - target = data[i + horizon]
    - last input = data[i]

    This makes ARIMA directly comparable to GRU and PC-GRU windowed inputs.
    """

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
    """
    Evaluate windowed ARIMA for one window-size / horizon setting.

    Parameters
    ----------
    window_size : int
        Number of past hours used as input history.

    horizon : int
        Number of hours ahead to forecast.

    Returns
    -------
    dict
        Validation metrics for the specified ARIMA-windowed configuration.
    """

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
    """
    Run windowed ARIMA evaluation across all configured settings.

    Results are saved to:

    outputs/metrics/arima_windowed_results.csv
    """

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