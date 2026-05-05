from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import (
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    VITAL_COLUMNS,
    ARIMA_ORDER,
    DROPOUT,
)

from src.models.gru_baseline import GRUBaseline
from src.models.pc_gru import PredictiveCodingGRU
from src.models.arima_baseline import ARIMABaseline


DATA_DIR = Path("data/processed/splits")
METRICS_DIR = Path("outputs/metrics")
CHECKPOINT_DIR = Path("outputs/checkpoints")
SCALER_PATH = DATA_DIR / "scaler.pkl"

VITAL_UNITS = {
    "heart_rate": "bpm",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "temperature": "°F",
    "respiratory_rate": "breaths/min",
    "oxygen_saturation": "%",
}

MODEL_FILES = {
    "GRU": METRICS_DIR / "gru_results.csv",
    "PC-GRU": METRICS_DIR / "pc_gru_results.csv",
    "ARIMA-windowed": METRICS_DIR / "arima_windowed_results.csv",
}


st.set_page_config(
    page_title="Sepsis Vital-Sign Forecasting Dashboard",
    layout="wide",
)


@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


def unnormalize_array(values: np.ndarray, scaler):
    if scaler is None:
        return values

    values = np.asarray(values)

    if values.ndim == 1:
        return values * scaler.scale_ + scaler.mean_

    return values * scaler.scale_ + scaler.mean_


def unnormalize_dataframe(df: pd.DataFrame, scaler) -> pd.DataFrame:
    if scaler is None:
        return df.copy()

    out = df.copy()
    out.loc[:, VITAL_COLUMNS] = unnormalize_array(
        out[VITAL_COLUMNS].values,
        scaler,
    )
    return out


@st.cache_data
def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    df["charttime"] = pd.to_datetime(df["charttime"])

    return df.sort_values(["subject_id", "hadm_id", "stay_id", "charttime"])


@st.cache_data
def load_metrics() -> pd.DataFrame:
    dfs = []

    for model_name, path in MODEL_FILES.items():
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if "epoch" in df.columns and "val_loss" in df.columns:
            df = df.loc[
                df.groupby(
                    ["model", "window_size_hours", "prediction_horizon_hours"]
                )["val_loss"].idxmin()
            ]

        df["model"] = model_name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


@st.cache_resource
def load_gru_model(window_size: int, horizon: int):
    path = CHECKPOINT_DIR / f"gru_w{window_size}_h{horizon}.pt"

    if not path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUBaseline(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model


@st.cache_resource
def load_pc_gru_model(window_size: int, horizon: int):
    path = CHECKPOINT_DIR / f"pc_gru_w{window_size}_h{horizon}.pt"

    if not path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PredictiveCodingGRU(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=2,
        dropout=DROPOUT,
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model


def get_patient_stays(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["subject_id", "hadm_id", "stay_id"]]
        .drop_duplicates()
        .sort_values(["subject_id", "hadm_id", "stay_id"])
    )


def make_stay_label(row) -> str:
    return f"Patient {row.subject_id} | HADM {row.hadm_id} | Stay {row.stay_id}"


def predict_gru(window_values_normalized: np.ndarray, window_size: int, horizon: int):
    model = load_gru_model(window_size, horizon)

    if model is None:
        return None

    device = next(model.parameters()).device
    x = torch.tensor(window_values_normalized, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).detach().cpu().numpy()[0]

    return pred


def predict_pc_gru(window_values_normalized: np.ndarray, window_size: int, horizon: int):
    model = load_pc_gru_model(window_size, horizon)

    if model is None:
        return None

    device = next(model.parameters()).device
    x = torch.tensor(window_values_normalized, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).detach().cpu().numpy()[0]

    return pred


def predict_arima(window_values_normalized: np.ndarray, horizon: int):
    model = ARIMABaseline(order=ARIMA_ORDER)
    preds = []

    for i in range(window_values_normalized.shape[1]):
        series = window_values_normalized[:, i]
        forecast = model.forecast_series(series, steps=horizon)
        preds.append(forecast[-1])

    return np.array(preds)


def get_all_predictions(window_values_normalized, window_size, horizon, selected_models):
    preds = {}

    if "GRU" in selected_models:
        pred = predict_gru(window_values_normalized, window_size, horizon)
        if pred is not None:
            preds["GRU"] = pred

    if "PC-GRU" in selected_models:
        pred = predict_pc_gru(window_values_normalized, window_size, horizon)
        if pred is not None:
            preds["PC-GRU"] = pred

    if "ARIMA-windowed" in selected_models:
        preds["ARIMA-windowed"] = predict_arima(window_values_normalized, horizon)

    return preds


def unnormalize_predictions(predictions_normalized, scaler):
    return {
        model_name: unnormalize_array(pred, scaler)
        for model_name, pred in predictions_normalized.items()
    }


def add_event_line(fig, event_time):
    fig.add_shape(
        type="line",
        x0=event_time,
        x1=event_time,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(dash="dash", width=2),
    )

    fig.add_annotation(
        x=event_time,
        y=1,
        xref="x",
        yref="paper",
        text="prediction point",
        showarrow=False,
        yanchor="bottom",
    )


def plot_single_vital(
    stay_df_display,
    vital,
    event_idx,
    window_size,
    horizon,
    predictions_display,
):
    start_idx = max(0, event_idx - window_size)
    true_future_idx = min(event_idx + horizon, len(stay_df_display) - 1)

    history_df = stay_df_display.iloc[start_idx : event_idx + 1]
    future_df = stay_df_display.iloc[event_idx:true_future_idx + 1]

    event_time = stay_df_display.iloc[event_idx]["charttime"]
    true_future_time = stay_df_display.iloc[true_future_idx]["charttime"]
    vital_idx = VITAL_COLUMNS.index(vital)
    unit = VITAL_UNITS.get(vital, "")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["charttime"],
            y=history_df[vital],
            mode="lines+markers",
            name="Observed history",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_df["charttime"],
            y=future_df[vital],
            mode="lines+markers",
            name="Ground truth future",
            line=dict(width=3, dash="dot"),
        )
    )

    for model_name, pred_vector in predictions_display.items():
        fig.add_trace(
            go.Scatter(
                x=[true_future_time],
                y=[pred_vector[vital_idx]],
                mode="markers",
                marker=dict(size=13, symbol="x"),
                name=f"{model_name} prediction",
            )
        )

    add_event_line(fig, event_time)

    fig.update_layout(
        height=520,
        title=f"{vital} forecast | Window={window_size}h | Horizon={horizon}h",
        xaxis_title="Time",
        yaxis_title=f"{vital} ({unit})" if unit else vital,
        hovermode="x unified",
        legend_title="Series",
        margin=dict(l=30, r=30, t=70, b=40),
    )

    return fig


def plot_mini_vital(
    stay_df_display,
    vital,
    event_idx,
    window_size,
    horizon,
    predictions_display,
):
    start_idx = max(0, event_idx - window_size)
    true_future_idx = min(event_idx + horizon, len(stay_df_display) - 1)

    history_df = stay_df_display.iloc[start_idx : event_idx + 1]
    future_df = stay_df_display.iloc[event_idx:true_future_idx + 1]

    event_time = stay_df_display.iloc[event_idx]["charttime"]
    true_future_time = stay_df_display.iloc[true_future_idx]["charttime"]
    vital_idx = VITAL_COLUMNS.index(vital)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["charttime"],
            y=history_df[vital],
            mode="lines",
            name="Observed",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_df["charttime"],
            y=future_df[vital],
            mode="lines",
            name="Ground truth",
        )
    )

    for model_name, pred_vector in predictions_display.items():
        fig.add_trace(
            go.Scatter(
                x=[true_future_time],
                y=[pred_vector[vital_idx]],
                mode="markers",
                marker=dict(size=10, symbol="x"),
                name=model_name,
            )
        )

    add_event_line(fig, event_time)

    unit = VITAL_UNITS.get(vital, "")

    fig.update_layout(
        height=300,
        title=f"{vital} ({unit})" if unit else vital,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=30),
    )

    return fig


def make_prediction_table(predictions_display, true_values_display):
    rows = []

    for model_name, pred in predictions_display.items():
        mae = float(np.mean(np.abs(pred - true_values_display)))
        rmse = float(np.sqrt(np.mean((pred - true_values_display) ** 2)))

        rows.append(
            {
                "model": model_name,
                "selected_window_mae": f"{mae:.3f}",
                "selected_window_rmse": f"{rmse:.3f}",
            }
        )

    return pd.DataFrame(rows)


def main():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0rem;
        }
        .subtitle {
            color: #666;
            margin-bottom: 1rem;
        }
        .sidebar-title {
            font-size: 1.2rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">Sepsis Vital-Sign Forecasting Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Predictive coding model vs ARIMA vs GRU on preset ICU cases</div>',
        unsafe_allow_html=True,
    )

    scaler = load_scaler()

    if scaler is None:
        st.warning(
            "Scaler file not found at data/processed/splits/scaler.pkl. "
            "Values will be displayed in normalized units."
        )

    tab_dashboard, tab_metrics, tab_mapping = st.tabs(
        ["Dashboard", "Model Comparison", "System Mapping"]
    )

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">Sepsis Forecast Demo</div>',
            unsafe_allow_html=True,
        )
        st.caption("Research dashboard using real saved model checkpoints")

        split_name = st.selectbox("Dataset split", ["val", "test", "train"], index=0)

        df = load_split(split_name)
        metrics_df = load_metrics()

        stays = get_patient_stays(df)
        stay_labels = [make_stay_label(row) for row in stays.itertuples(index=False)]

        selected_label = st.selectbox("Select preset patient / ICU stay", stay_labels)
        selected_idx = stay_labels.index(selected_label)
        selected_stay = stays.iloc[selected_idx]

        stay_df = df[
            (df["subject_id"] == selected_stay["subject_id"])
            & (df["hadm_id"] == selected_stay["hadm_id"])
            & (df["stay_id"] == selected_stay["stay_id"])
        ].sort_values("charttime").reset_index(drop=True)

        stay_df_display = unnormalize_dataframe(stay_df, scaler)

        window_size = st.selectbox("Observed window before event", [2, 12, 24], index=1)
        horizon = st.selectbox("Forecast horizon", [1, 2, 4], index=2)

        selected_vital = st.selectbox("Main vital sign", VITAL_COLUMNS)

        selected_panel_vitals = st.multiselect(
            "Separate vital panels",
            VITAL_COLUMNS,
            default=["heart_rate", "respiratory_rate", "temperature"],
        )

        selected_models = st.multiselect(
            "Models to display",
            ["GRU", "PC-GRU", "ARIMA-windowed"],
            default=["GRU", "PC-GRU", "ARIMA-windowed"],
        )

    if len(stay_df) <= window_size + horizon:
        st.error("This ICU stay is too short for the selected window and horizon.")
        return

    min_event_idx = window_size
    max_event_idx = len(stay_df) - horizon - 1
    event_options = list(range(min_event_idx, max_event_idx + 1))

    with st.sidebar:
        event_idx = st.select_slider(
            "Event / prediction time",
            options=event_options,
            value=event_options[len(event_options) // 2],
            format_func=lambda i: str(stay_df.iloc[i]["charttime"]),
        )

    start_idx = event_idx - window_size
    true_future_idx = event_idx + horizon

    # Normalized values are used for model input.
    window_values_normalized = (
        stay_df.iloc[start_idx:event_idx][VITAL_COLUMNS]
        .values
        .astype(np.float32)
    )

    true_values_normalized = (
        stay_df.iloc[true_future_idx][VITAL_COLUMNS]
        .values
        .astype(np.float32)
    )

    predictions_normalized = get_all_predictions(
        window_values_normalized=window_values_normalized,
        window_size=window_size,
        horizon=horizon,
        selected_models=selected_models,
    )

    # Un-normalized values are used for plots and local error table.
    true_values_display = unnormalize_array(true_values_normalized, scaler)
    predictions_display = unnormalize_predictions(predictions_normalized, scaler)

    with tab_dashboard:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Selected Case",
            f"Patient {selected_stay['subject_id']}",
            f"Stay {selected_stay['stay_id']}",
        )
        col2.metric("Forecast Horizon", f"Next {horizon} hour(s)")
        col3.metric("Observed Window", f"{window_size} hour(s)")

        left_col, right_col = st.columns([2.2, 1])

        with left_col:
            st.subheader("Future vital-sign forecast")

            fig = plot_single_vital(
                stay_df_display=stay_df_display,
                vital=selected_vital,
                event_idx=event_idx,
                window_size=window_size,
                horizon=horizon,
                predictions_display=predictions_display,
            )

            st.plotly_chart(fig, use_container_width=True)

        with right_col:
            st.subheader("Selected-window prediction error")

            if not predictions_display:
                st.warning(
                    "No model predictions available. Check that checkpoints exist in outputs/checkpoints."
                )
            else:
                pred_table = make_prediction_table(
                    predictions_display,
                    true_values_display,
                )
                st.dataframe(pred_table, use_container_width=True, hide_index=True)

            st.subheader("Checkpoint status")
            status_rows = [
                {
                    "model": "GRU",
                    "checkpoint": str(CHECKPOINT_DIR / f"gru_w{window_size}_h{horizon}.pt"),
                    "found": (CHECKPOINT_DIR / f"gru_w{window_size}_h{horizon}.pt").exists(),
                },
                {
                    "model": "PC-GRU",
                    "checkpoint": str(CHECKPOINT_DIR / f"pc_gru_w{window_size}_h{horizon}.pt"),
                    "found": (CHECKPOINT_DIR / f"pc_gru_w{window_size}_h{horizon}.pt").exists(),
                },
                {
                    "model": "ARIMA-windowed",
                    "checkpoint": "fit live from selected window",
                    "found": True,
                },
            ]
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        st.subheader("Separate vital panels")

        if selected_panel_vitals:
            panel_cols = st.columns(min(3, len(selected_panel_vitals)))

            for i, vital in enumerate(selected_panel_vitals):
                with panel_cols[i % len(panel_cols)]:
                    mini_fig = plot_mini_vital(
                        stay_df_display=stay_df_display,
                        vital=vital,
                        event_idx=event_idx,
                        window_size=window_size,
                        horizon=horizon,
                        predictions_display=predictions_display,
                    )
                    st.plotly_chart(mini_fig, use_container_width=True)

        st.subheader("Selected raw window and future target")

        display_cols = ["charttime"] + VITAL_COLUMNS
        display_df = stay_df_display.iloc[start_idx:true_future_idx + 1][display_cols].copy()

        for col in VITAL_COLUMNS:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True)

    with tab_metrics:
        st.subheader("Saved model summary metrics")

        if metrics_df.empty:
            st.warning("No saved model metric files found in outputs/metrics.")
        else:
            subset = metrics_df[
                (metrics_df["window_size_hours"] == window_size)
                & (metrics_df["prediction_horizon_hours"] == horizon)
            ]

            if subset.empty:
                st.warning("No saved metrics available for this window/horizon.")
            else:
                metric_cols = [
                    "model",
                    "window_size_hours",
                    "prediction_horizon_hours",
                    "mae",
                    "rmse",
                    "pearson",
                    "direction_accuracy",
                ]
                metric_cols = [c for c in metric_cols if c in subset.columns]
                display_subset = subset[metric_cols].copy()

                for col in ["mae", "rmse", "pearson", "direction_accuracy"]:
                    if col in display_subset.columns:
                        display_subset[col] = display_subset[col].map(lambda x: f"{x:.3f}")

                st.dataframe(display_subset, use_container_width=True, hide_index=True)

    with tab_mapping:
        st.subheader("System mapping")

        st.markdown(
            """
            **Pipeline used by this dashboard:**

            1. Load processed ICU split file from `data/processed/splits/`
            2. Load the saved training scaler from `data/processed/splits/scaler.pkl`
            3. Select a patient ICU stay
            4. Select an observed window before the prediction point
            5. Send normalized values into GRU and PC-GRU checkpoints
            6. Fit ARIMA live on the selected normalized observed window
            7. Un-normalize predictions and true values back into clinical units
            8. Compare predictions against the true future vital signs

            **Displayed vital signs:**

            - Heart rate
            - Systolic blood pressure
            - Diastolic blood pressure
            - Temperature
            - Respiratory rate
            - Oxygen saturation
            """
        )


if __name__ == "__main__":
    main()
    