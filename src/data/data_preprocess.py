from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Vital-sign columns used throughout the project
VITAL_COLUMNS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "oxygen_saturation",
]


def load_vitals(raw_data_dir):
    """
    Load raw MIMIC-IV ICU chart events and item metadata.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the MIMIC-IV demo dataset directory.

    Returns
    -------
    pandas.DataFrame
        Merged dataframe containing chart events with human-readable labels.

    Notes
    -----
    - chartevents.csv.gz contains raw measurements.
    - d_items.csv.gz maps item IDs to labels (e.g., "Heart Rate").
    - The merge attaches labels to each measurement.
    """

    raw_data_dir = Path(raw_data_dir)

    chartevents_path = raw_data_dir / "icu" / "chartevents.csv.gz"
    d_items_path = raw_data_dir / "icu" / "d_items.csv.gz"

    chartevents = pd.read_csv(chartevents_path)
    d_items = pd.read_csv(d_items_path)

    # Attach descriptive labels to each measurement
    return chartevents.merge(
        d_items[["itemid", "label"]],
        on="itemid",
        how="left",
    )


def extract_vitals(df):
    """
    Extract and reshape vital signs into a structured time-series format.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw chart events dataframe with labels.

    Returns
    -------
    pandas.DataFrame
        Wide-format dataframe with one row per timestamp and columns for each vital sign.

    Notes
    -----
    - Filters only relevant vital signs.
    - Converts long-format data to wide-format using pivot.
    - Handles temperature conversion from Celsius to Fahrenheit.
    """

    # Mapping from MIMIC labels to standardized column names
    vital_map = {
        "Heart Rate": "heart_rate",
        "Respiratory Rate": "respiratory_rate",
        "O2 saturation pulseoxymetry": "oxygen_saturation",
        "Temperature Fahrenheit": "temperature_f",
        "Temperature Celsius": "temperature_c",
        "Non Invasive Blood Pressure systolic": "systolic_bp",
        "Non Invasive Blood Pressure diastolic": "diastolic_bp",
    }

    # Keep only relevant vital signs
    df = df[df["label"].isin(vital_map.keys())].copy()
    df["vital_name"] = df["label"].map(vital_map)

    # Select relevant columns
    df = df[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "vital_name",
            "valuenum",
        ]
    ]

    df["charttime"] = pd.to_datetime(df["charttime"])

    # Convert from long format to wide format (one column per vital sign)
    wide = df.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "charttime"],
        columns="vital_name",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()

    # Convert Celsius to Fahrenheit and unify temperature column
    if "temperature_c" in wide.columns:
        wide["temperature_c"] = (wide["temperature_c"] * 9 / 5) + 32

        if "temperature_f" in wide.columns:
            wide["temperature_f"] = wide["temperature_f"].fillna(
                wide["temperature_c"]
            )
        else:
            wide["temperature_f"] = wide["temperature_c"]

        wide = wide.drop(columns=["temperature_c"])

    if "temperature_f" in wide.columns:
        wide = wide.rename(columns={"temperature_f": "temperature"})

    return wide


def resample_and_fill(vitals):
    """
    Resample time series to uniform hourly intervals and forward-fill missing values.

    Parameters
    ----------
    vitals : pandas.DataFrame
        Wide-format vital-sign dataframe.

    Returns
    -------
    pandas.DataFrame
        Resampled and cleaned dataframe.

    Notes
    -----
    - Resampling ensures consistent 1-hour intervals across patients.
    - Forward fill is applied within each ICU stay.
    - Backward fill is intentionally avoided to prevent data leakage.
    """

    # Sort data to ensure correct temporal order
    vitals = vitals.sort_values(
        ["subject_id", "hadm_id", "stay_id", "charttime"]
    )

    vitals = vitals.set_index("charttime")

    # Resample each ICU stay to 1-hour intervals using mean aggregation
    vitals = (
        vitals
        .groupby(["subject_id", "hadm_id", "stay_id"])
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )

    # Forward fill missing values within each ICU stay
    vitals[VITAL_COLUMNS] = (
        vitals
        .groupby(["subject_id", "hadm_id", "stay_id"])[VITAL_COLUMNS]
        .ffill()
    )

    # Drop rows where vital signs are still missing after forward fill
    return vitals.dropna(subset=VITAL_COLUMNS)


def split_and_normalize(vitals, output_dir, seed=42):
    """
    Split dataset into train/validation/test sets and apply normalization.

    Parameters
    ----------
    vitals : pandas.DataFrame
        Preprocessed vital-sign dataset.

    output_dir : str or pathlib.Path
        Directory where split datasets and scaler will be saved.

    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Notes
    -----
    - Splitting is performed at the patient level to avoid leakage.
    - StandardScaler is fit on training data only.
    - The same scaler is applied to validation and test sets.
    """

    subject_ids = vitals["subject_id"].unique()

    # Split patients into train/validation/test
    train_ids, temp_ids = train_test_split(
        subject_ids,
        test_size=0.30,
        random_state=seed,
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=seed,
    )

    train_df = vitals[vitals["subject_id"].isin(train_ids)].copy()
    val_df = vitals[vitals["subject_id"].isin(val_ids)].copy()
    test_df = vitals[vitals["subject_id"].isin(test_ids)].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit scaler only on training data to avoid leakage
    scaler = StandardScaler()
    scaler.fit(train_df[VITAL_COLUMNS])

    # Save scaler for later inverse transformation (e.g., dashboard)
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Apply normalization
    train_df.loc[:, VITAL_COLUMNS] = scaler.transform(train_df[VITAL_COLUMNS])
    val_df.loc[:, VITAL_COLUMNS] = scaler.transform(val_df[VITAL_COLUMNS])
    test_df.loc[:, VITAL_COLUMNS] = scaler.transform(test_df[VITAL_COLUMNS])

    # Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)
    print("Saved scaler:", scaler_path)


def main():
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Load raw MIMIC-IV data
    2. Extract vital signs
    3. Resample and clean time series
    4. Split into train/validation/test sets
    5. Normalize and save outputs
    """

    raw = load_vitals("data/raw/mimic-iv-demo")
    vitals = extract_vitals(raw)
    vitals = resample_and_fill(vitals)

    split_and_normalize(
        vitals,
        output_dir="data/processed/splits",
    )


if __name__ == "__main__":
    main()