
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


VITAL_COLUMNS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "oxygen_saturation",
]


def load_vitals(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)

    chartevents_path = raw_data_dir / "icu" / "chartevents.csv.gz"
    d_items_path = raw_data_dir / "icu" / "d_items.csv.gz"

    chartevents = pd.read_csv(chartevents_path)
    d_items = pd.read_csv(d_items_path)

    return chartevents.merge(
        d_items[["itemid", "label"]],
        on="itemid",
        how="left",
    )


def extract_vitals(df):
    vital_map = {
        "Heart Rate": "heart_rate",
        "Respiratory Rate": "respiratory_rate",
        "O2 saturation pulseoxymetry": "oxygen_saturation",
        "Temperature Fahrenheit": "temperature_f",
        "Temperature Celsius": "temperature_c",
        "Non Invasive Blood Pressure systolic": "systolic_bp",
        "Non Invasive Blood Pressure diastolic": "diastolic_bp",
    }

    df = df[df["label"].isin(vital_map.keys())].copy()
    df["vital_name"] = df["label"].map(vital_map)

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

    wide = df.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "charttime"],
        columns="vital_name",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()

    # Celsius → Fahrenheit
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
    vitals = vitals.sort_values(
        ["subject_id", "hadm_id", "stay_id", "charttime"]
    )

    vitals = vitals.set_index("charttime")

    vitals = (
        vitals
        .groupby(["subject_id", "hadm_id", "stay_id"])
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )

    # Forward fill only to avoid using future values for earlier time points.
    # Fill is restricted within each ICU stay.
    vitals[VITAL_COLUMNS] = (
        vitals
        .groupby(["subject_id", "hadm_id", "stay_id"])[VITAL_COLUMNS]
        .ffill()
    )

    return vitals.dropna(subset=VITAL_COLUMNS)


def split_and_normalize(vitals, output_dir, seed=42):
    subject_ids = vitals["subject_id"].unique()

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

    # Fit scaler on training data only to avoid leakage.
    scaler = StandardScaler()
    scaler.fit(train_df[VITAL_COLUMNS])

    # Save scaler so dashboard/results can un-normalize values back to clinical units.
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    train_df.loc[:, VITAL_COLUMNS] = scaler.transform(train_df[VITAL_COLUMNS])
    val_df.loc[:, VITAL_COLUMNS] = scaler.transform(val_df[VITAL_COLUMNS])
    test_df.loc[:, VITAL_COLUMNS] = scaler.transform(test_df[VITAL_COLUMNS])

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)
    print("Saved scaler:", scaler_path)


def main():
    raw = load_vitals("data/raw/mimic-iv-demo")
    vitals = extract_vitals(raw)
    vitals = resample_and_fill(vitals)

    split_and_normalize(
        vitals,
        output_dir="data/processed/splits",
    )


if __name__ == "__main__":
    main()
