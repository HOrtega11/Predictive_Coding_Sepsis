from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Directory structure for inputs and outputs
METRICS_DIR = Path("outputs/metrics")
PLOTS_DIR = Path("outputs/plots")
TABLES_DIR = Path("outputs/tables")

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_final_epoch(path, model_name):
    """
    Load model results and select the best epoch per configuration.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file containing model results.

    model_name : str
        Name of the model (e.g., "GRU", "PC-GRU").

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe containing one row per
        (model, window_size, horizon) combination.

    Notes
    -----
    - If validation loss is available, the epoch with the lowest validation loss is selected.
    - Otherwise, the final epoch is used.
    """

    df = pd.read_csv(path)

    if "epoch" in df.columns:
        if "val_loss" in df.columns:
            # Select epoch with minimum validation loss
            df = df.loc[
                df.groupby(
                    ["model", "window_size_hours", "prediction_horizon_hours"]
                )["val_loss"].idxmin()
            ]
        else:
            # Fallback: select final epoch
            df = df.sort_values("epoch")
            df = df.groupby(
                ["model", "window_size_hours", "prediction_horizon_hours"],
                as_index=False,
            ).tail(1)

    df["model"] = model_name
    return df


def load_results():
    """
    Load and combine results from all available models.

    Returns
    -------
    pandas.DataFrame
        Combined dataframe containing GRU, PC-GRU, and ARIMA results.

    Raises
    ------
    FileNotFoundError
        If no result CSV files are found.
    """

    dfs = []

    gru_path = METRICS_DIR / "gru_results.csv"
    pc_path = METRICS_DIR / "pc_gru_results.csv"
    arima_windowed_path = METRICS_DIR / "arima_windowed_results.csv"

    if gru_path.exists():
        dfs.append(load_final_epoch(gru_path, "GRU"))

    if pc_path.exists():
        dfs.append(load_final_epoch(pc_path, "PC-GRU"))

    # ARIMA does not have epochs, so load directly
    if arima_windowed_path.exists():
        dfs.append(pd.read_csv(arima_windowed_path))

    if not dfs:
        raise FileNotFoundError("No result CSV files found in outputs/metrics/")

    return pd.concat(dfs, ignore_index=True)


def make_summary_table(df):
    """
    Create and save a summary table of model performance.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined results dataframe.

    Notes
    -----
    The table includes key metrics for all configurations.
    """

    cols = [
        "model",
        "window_size_hours",
        "prediction_horizon_hours",
        "mae",
        "rmse",
        "pearson",
        "direction_accuracy",
    ]

    summary = df[cols].sort_values(
        ["prediction_horizon_hours", "window_size_hours", "model"]
    )

    summary.to_csv(TABLES_DIR / "model_summary_table.csv", index=False)
    print("\nSaved:", TABLES_DIR / "model_summary_table.csv")
    print(summary)


def make_best_model_table(df):
    """
    Identify the best model per metric and configuration.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined results dataframe.

    Notes
    -----
    - MAE and RMSE are minimized.
    - Pearson and directional accuracy are maximized.
    """

    best_rows = []

    # Metrics where lower is better
    for metric in ["mae", "rmse"]:
        best = df.loc[
            df.groupby(
                ["window_size_hours", "prediction_horizon_hours"]
            )[metric].idxmin()
        ].copy()
        best["best_by"] = metric
        best_rows.append(best)

    # Metrics where higher is better
    for metric in ["pearson", "direction_accuracy"]:
        best = df.loc[
            df.groupby(
                ["window_size_hours", "prediction_horizon_hours"]
            )[metric].idxmax()
        ].copy()
        best["best_by"] = metric
        best_rows.append(best)

    best_df = pd.concat(best_rows, ignore_index=True)

    best_df = best_df[
        [
            "best_by",
            "model",
            "window_size_hours",
            "prediction_horizon_hours",
            "mae",
            "rmse",
            "pearson",
            "direction_accuracy",
        ]
    ]

    best_df.to_csv(TABLES_DIR / "best_model_by_metric.csv", index=False)
    print("\nSaved:", TABLES_DIR / "best_model_by_metric.csv")


def plot_metric_by_horizon(df, metric):
    """
    Plot a metric as a function of prediction horizon.

    Parameters
    ----------
    df : pandas.DataFrame
        Results dataframe.

    metric : str
        Metric to plot (e.g., "mae", "rmse").
    """

    for window_size in sorted(df["window_size_hours"].unique()):
        subset = df[df["window_size_hours"] == window_size]

        plt.figure(figsize=(8, 5))

        for model_name, model_df in subset.groupby("model"):
            model_df = model_df.sort_values("prediction_horizon_hours")

            plt.plot(
                model_df["prediction_horizon_hours"],
                model_df[metric],
                marker="o",
                label=model_name,
            )

        plt.xlabel("Prediction Horizon (hours)")
        plt.ylabel(metric)
        plt.title(f"{metric} by Horizon | Window={window_size}h")
        plt.legend()
        plt.tight_layout()

        output_path = PLOTS_DIR / f"{metric}_by_horizon_window_{window_size}.png"
        plt.savefig(output_path)
        plt.close()

        print("Saved:", output_path)


def plot_metric_by_window(df, metric):
    """
    Plot a metric as a function of input window size.

    Parameters
    ----------
    df : pandas.DataFrame
        Results dataframe.

    metric : str
        Metric to plot.
    """

    for horizon in sorted(df["prediction_horizon_hours"].unique()):
        subset = df[df["prediction_horizon_hours"] == horizon]

        plt.figure(figsize=(8, 5))

        for model_name, model_df in subset.groupby("model"):
            model_df = model_df.sort_values("window_size_hours")

            plt.plot(
                model_df["window_size_hours"],
                model_df[metric],
                marker="o",
                label=model_name,
            )

        plt.xlabel("Input Window Size (hours)")
        plt.ylabel(metric)
        plt.title(f"{metric} by Window Size | Horizon={horizon}h")
        plt.legend()
        plt.tight_layout()

        output_path = PLOTS_DIR / f"{metric}_by_window_horizon_{horizon}.png"
        plt.savefig(output_path)
        plt.close()

        print("Saved:", output_path)


def make_per_variable_table(df):
    """
    Create a table of per-variable MAE values.

    Parameters
    ----------
    df : pandas.DataFrame
        Results dataframe.
    """

    variable_cols = [
        col for col in df.columns
        if col.startswith("mae_")
        and col not in ["mae"]
    ]

    cols = [
        "model",
        "window_size_hours",
        "prediction_horizon_hours",
    ] + variable_cols

    per_var = df[cols].sort_values(
        ["prediction_horizon_hours", "window_size_hours", "model"]
    )

    per_var.to_csv(TABLES_DIR / "per_variable_mae_table.csv", index=False)
    print("\nSaved:", TABLES_DIR / "per_variable_mae_table.csv")


def plot_per_variable_mae(df, window_size=24, horizon=4):
    """
    Plot per-variable MAE as a bar chart for a specific configuration.

    Parameters
    ----------
    df : pandas.DataFrame
        Results dataframe.

    window_size : int
        Input window size.

    horizon : int
        Prediction horizon.
    """

    variable_cols = [
        col for col in df.columns
        if col.startswith("mae_")
        and col not in ["mae"]
    ]

    subset = df[
        (df["window_size_hours"] == window_size)
        & (df["prediction_horizon_hours"] == horizon)
    ]

    if subset.empty:
        print(f"Skipping per-variable MAE plot; no data for W={window_size}, H={horizon}")
        return

    plot_df = subset[["model"] + variable_cols].set_index("model").T
    plot_df.index = [name.replace("mae_", "") for name in plot_df.index]

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_xlabel("Vital Sign")
    ax.set_ylabel("MAE")
    ax.set_title(f"Per-Variable MAE | Window={window_size}h, Horizon={horizon}h")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = PLOTS_DIR / f"per_variable_mae_w{window_size}_h{horizon}.png"
    plt.savefig(output_path)
    plt.close()

    print("Saved:", output_path)


def plot_loss_curves(csv_path, model_name):
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the metrics CSV file.

    model_name : str
        Name of the model (e.g., "GRU", "PC-GRU").
    """

    if not csv_path.exists():
        print(f"Skipping loss curves; missing {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        print(f"Skipping loss curves for {model_name}; no epoch column")
        return

    for loss_type in ["train_loss", "val_loss"]:
        plt.figure(figsize=(10, 6))

        for (window, horizon), group in df.groupby(
            ["window_size_hours", "prediction_horizon_hours"]
        ):
            group = group.sort_values("epoch")
            label = f"W={window}h, H={horizon}h"

            plt.plot(
                group["epoch"],
                group[loss_type],
                marker="o",
                label=label,
            )

        plt.xlabel("Epoch")
        plt.ylabel(loss_type.replace("_", " ").title())
        plt.title(f"{model_name} {loss_type.replace('_', ' ').title()}")
        plt.legend()
        plt.tight_layout()

        output_path = (
            PLOTS_DIR
            / f"{model_name.lower().replace('-', '_')}_{loss_type}.png"
        )

        plt.savefig(output_path)
        plt.close()

        print("Saved:", output_path)


def main():
    """
    Generate all result tables and plots.

    Outputs
    -------
    Tables:
        outputs/tables/

    Plots:
        outputs/plots/
    """

    df = load_results()

    make_summary_table(df)
    make_best_model_table(df)
    make_per_variable_table(df)

    for metric in ["mae", "rmse", "pearson", "direction_accuracy"]:
        plot_metric_by_horizon(df, metric)
        plot_metric_by_window(df, metric)

    plot_per_variable_mae(df, window_size=2, horizon=1)
    plot_per_variable_mae(df, window_size=2, horizon=2)
    plot_per_variable_mae(df, window_size=24, horizon=4)
    plot_per_variable_mae(df, window_size=12, horizon=4)

    plot_loss_curves(METRICS_DIR / "gru_results.csv", "GRU")
    plot_loss_curves(METRICS_DIR / "pc_gru_results.csv", "PC-GRU")

    print("\nDone. Tables saved to outputs/tables/ and plots saved to outputs/plots/")


if __name__ == "__main__":
    main()