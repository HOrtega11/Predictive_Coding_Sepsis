
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


DATA_DIR = Path("data/processed/splits")
METRICS_DIR = Path("outputs/metrics")

VITAL_COLUMNS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "oxygen_saturation",
]

MODEL_FILES = {
    "GRU": METRICS_DIR / "gru_results.csv",
    "PC-GRU": METRICS_DIR / "pc_gru_results.csv",
    "ARIMA-windowed": METRICS_DIR / "arima_windowed_results.csv",
}


@st.cache_data
def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}.csv"
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


def get_patient_stays(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["subject_id", "hadm_id", "stay_id"]]
        .drop_duplicates()
        .sort_values(["subject_id", "hadm_id", "stay_id"])
    )


def make_stay_label(row) -> str:
    return f"Patient {row.subject_id} | HADM {row.hadm_id} | Stay {row.stay_id}"


def estimate_prediction(
    window_values: np.ndarray,
    model_name: str,
    vital: str,
    horizon: int,
) -> float:
    """
    Lightweight dashboard approximation.

    This is not final model inference. It lets the dashboard show model behavior
    without requiring saved model checkpoints. Replace this with real checkpoint
    loading if needed.
    """
    last_value = window_values[-1]
    trend = window_values[-1] - window_values[0]

    if model_name == "ARIMA-windowed":
        return last_value + 0.25 * trend

    if model_name == "GRU":
        return last_value + 0.35 * trend

    if model_name == "PC-GRU":
        return last_value + 0.30 * trend

    return last_value


def plot_vital_window(
    stay_df: pd.DataFrame,
    vital: str,
    event_idx: int,
    window_size: int,
    horizon: int,
    selected_models: list[str],
):
    start_idx = max(0, event_idx - window_size)
    end_idx = event_idx + horizon

    plot_df = stay_df.iloc[start_idx : min(end_idx + 1, len(stay_df))].copy()
    window_df = stay_df.iloc[start_idx:event_idx].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["charttime"],
            y=plot_df[vital],
            mode="lines+markers",
            name="Observed",
        )
    )

    event_time = stay_df.iloc[event_idx]["charttime"]
    true_future_idx = min(event_idx + horizon, len(stay_df) - 1)
    true_future_time = stay_df.iloc[true_future_idx]["charttime"]
    true_future_value = stay_df.iloc[true_future_idx][vital]

    fig.add_shape(
        type="line",
        x0=event_time,
        x1=event_time,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(dash="dash"),
    )

    fig.add_annotation(
        x=event_time,
        y=1,
        xref="x",
        yref="paper",
        text="Event / prediction point",
        showarrow=False,
        yanchor="bottom",
    )

    fig.add_trace(
        go.Scatter(
            x=[true_future_time],
            y=[true_future_value],
            mode="markers",
            marker=dict(size=12),
            name=f"True +{horizon}h",
        )
    )

    if len(window_df) > 1:
        window_values = window_df[vital].values

        for model_name in selected_models:
            pred = estimate_prediction(
                window_values=window_values,
                model_name=model_name,
                vital=vital,
                horizon=horizon,
            )

            fig.add_trace(
                go.Scatter(
                    x=[true_future_time],
                    y=[pred],
                    mode="markers",
                    marker=dict(size=11, symbol="x"),
                    name=f"{model_name} prediction",
                )
            )

    fig.update_layout(
        title=f"{vital} | Window={window_size}h | Horizon={horizon}h",
        xaxis_title="Time",
        yaxis_title="Standardized value",
        hovermode="x unified",
    )

    return fig


def main():
    st.set_page_config(
        page_title="Predictive Coding Sepsis Dashboard",
        layout="wide",
    )

    st.title("Predictive Coding Sepsis Forecasting Dashboard")

    st.markdown(
        """
        This dashboard lets you inspect ICU vital-sign windows before a selected
        event time and compare approximate model predictions against the actual
        future value. The displayed values are standardized because the processed
        split files are normalized.
        """
    )

    split_name = st.sidebar.selectbox(
        "Dataset split",
        ["val", "test", "train"],
        index=0,
    )

    df = load_split(split_name)
    metrics_df = load_metrics()

    stays = get_patient_stays(df)
    stay_labels = [make_stay_label(row) for row in stays.itertuples(index=False)]

    selected_label = st.sidebar.selectbox("Preset patient / ICU stay", stay_labels)
    selected_idx = stay_labels.index(selected_label)
    selected_stay = stays.iloc[selected_idx]

    stay_df = df[
        (df["subject_id"] == selected_stay["subject_id"])
        & (df["hadm_id"] == selected_stay["hadm_id"])
        & (df["stay_id"] == selected_stay["stay_id"])
    ].sort_values("charttime")

    window_size = st.sidebar.selectbox("Input window before event", [2, 12, 24])
    horizon = st.sidebar.selectbox("Prediction horizon", [1, 2, 4])

    vital = st.sidebar.selectbox("Vital sign", VITAL_COLUMNS)

    available_models = ["GRU", "PC-GRU", "ARIMA-windowed"]
    selected_models = st.sidebar.multiselect(
        "Models to display",
        available_models,
        default=available_models,
    )

    min_event_idx = window_size
    max_event_idx = max(window_size, len(stay_df) - horizon - 1)

    if len(stay_df) <= window_size + horizon:
        st.error("This ICU stay is too short for the selected window and horizon.")
        return

    event_options = list(range(min_event_idx, max_event_idx + 1))

    event_idx = st.sidebar.select_slider(
        "Event / prediction time",
        options=event_options,
        value=event_options[len(event_options) // 2],
        format_func=lambda i: str(stay_df.iloc[i]["charttime"]),
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Patient ID", selected_stay["subject_id"])
    col2.metric("ICU Stay ID", selected_stay["stay_id"])
    col3.metric("Rows in stay", len(stay_df))

    st.subheader("Vital Forecast View")

    fig = plot_vital_window(
        stay_df=stay_df,
        vital=vital,
        event_idx=event_idx,
        window_size=window_size,
        horizon=horizon,
        selected_models=selected_models,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Window Data")

    start_idx = max(0, event_idx - window_size)
    true_future_idx = min(event_idx + horizon, len(stay_df) - 1)

    display_cols = ["charttime"] + VITAL_COLUMNS
    st.dataframe(
        stay_df.iloc[start_idx : true_future_idx + 1][display_cols],
        use_container_width=True,
    )

    st.subheader("Model Summary Metrics")

    if metrics_df.empty:
        st.warning("No saved model metric files found in outputs/metrics.")
    else:
        subset = metrics_df[
            (metrics_df["window_size_hours"] == window_size)
            & (metrics_df["prediction_horizon_hours"] == horizon)
        ]

        if subset.empty:
            st.warning("No metrics available for the selected window/horizon.")
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
            st.dataframe(subset[metric_cols], use_container_width=True)

    st.info(
        """
        Note: This dashboard uses processed split files and saved result CSVs.
        The plotted model predictions are lightweight approximations for
        interactive visualization unless model checkpoints and per-patient
        prediction files are added.
        """
    )


if __name__ == "__main__":
    main()
