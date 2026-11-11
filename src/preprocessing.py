import numpy as np


def create_period_sequences(
    data,
    city_keys,
    target_start,
    target_end,
    input_length=60,
    horizon=12
):
    X, y = [], []

    for _, city_data in data.groupby(city_keys, sort=False):
        city_data = city_data.sort_values("dt").reset_index(drop=True)

        values = city_data["temperature_anomaly"].to_numpy(
            dtype=np.float32
        )
        dates = city_data["dt"].to_numpy()

        for i in range(len(values) - input_length - horizon + 1):
            target_dates = dates[
                i + input_length:i + input_length + horizon
            ]

            if (
                target_dates[0] >= np.datetime64(target_start)
                and target_dates[-1] <= np.datetime64(target_end)
            ):
                X.append(values[i:i + input_length])
                y.append(
                    values[i + input_length:i + input_length + horizon]
                )

    return (
        np.asarray(X, dtype=np.float32)[..., np.newaxis],
        np.asarray(y, dtype=np.float32)
    )

def calculate_temperature_anomalies(train_df, val_df, test_df, city_keys):
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()

    monthly_reference = (
        train.groupby(city_keys + ["month"])["AverageTemperature"]
        .mean()
        .rename("reference_temperature")
        .reset_index()
    )

    results = []

    for data in [train, val, test]:
        data = data.drop(
            columns=["reference_temperature", "temperature_anomaly"],
            errors="ignore"
        )

        data = data.merge(
            monthly_reference,
            on=city_keys + ["month"],
            how="left",
            validate="many_to_one"
        )

        if data["reference_temperature"].isna().any():
            raise ValueError("Bazı şehir-ay kombinasyonları için referans bulunamadı.")

        data["temperature_anomaly"] = (
            data["AverageTemperature"] - data["reference_temperature"]
        )

        results.append(data)

    return (*results, monthly_reference)

import pandas as pd


def load_temperature_data(
    csv_path,
    start_date="1900-01-01",
    end_date="2012-12-01"
):
    df = pd.read_csv(
        csv_path,
        parse_dates=["dt"]
    )

    required_columns = [
        "dt",
        "AverageTemperature",
        "City",
        "Country",
        "Latitude",
        "Longitude"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {missing_columns}"
        )

    df = df.loc[
        df["dt"].between(start_date, end_date)
    ].copy()

    df["month"] = df["dt"].dt.month

    return df

def convert_coordinates(df):
    data = df.copy()

    data["lat"] = (
        data["Latitude"].str[:-1].astype(float)
        * data["Latitude"].str[-1].map({"N": 1, "S": -1})
    )

    data["lon"] = (
        data["Longitude"].str[:-1].astype(float)
        * data["Longitude"].str[-1].map({"E": 1, "W": -1})
    )

    if data[["lat", "lon"]].isna().any().any():
        raise ValueError("Geçersiz koordinat bulundu.")

    return data

def calculate_city_quality(df):
    city_keys = ["City", "Country", "Latitude", "Longitude"]

    quality = (
        df.groupby(city_keys, as_index=False)
        .agg(
            total_records=("dt", "size"),
            valid_records=("AverageTemperature", "count"),
            first_date=("dt", "min"),
            last_date=("dt", "max"),
            lat=("lat", "first"),
            lon=("lon", "first")
        )
    )

    expected_months = 113 * 12

    quality["coverage_pct"] = (
        quality["valid_records"] / expected_months * 100
    )

    quality["latitude_band"] = pd.cut(
        quality["lat"],
        bins=[-90, -60, -30, 0, 30, 60, 90],
        labels=[
            "60–90° Güney",
            "30–60° Güney",
            "0–30° Güney",
            "0–30° Kuzey",
            "30–60° Kuzey",
            "60–90° Kuzey"
        ],
        include_lowest=True
    )

    return quality

def select_cities(quality, n_cities=200, random_state=42):
    quotas = {
        "30–60° Güney": 35,
        "0–30° Güney": 40,
        "0–30° Kuzey": 40,
        "30–60° Kuzey": 60,
        "60–90° Kuzey": 25
    }

    eligible = quality[
        quality["coverage_pct"] == 100
    ].copy()

    eligible = eligible.drop_duplicates(
        subset=["lat", "lon"]
    )

    selected = []

    for band, quota in quotas.items():
        candidates = eligible[
            eligible["latitude_band"] == band
        ]

        n = min(quota, len(candidates))

        sampled = candidates.sample(
            n=n,
            random_state=random_state
        )

        selected.append(sampled)

    selected_final = pd.concat(
        selected,
        ignore_index=True
    )

    remaining = n_cities - len(selected_final)

    if remaining > 0:
        available = eligible[
            ~eligible.index.isin(selected_final.index)
        ]

        available = available[
            ~available.set_index(["lat", "lon"]).index.isin(
                selected_final.set_index(["lat", "lon"]).index
            )
        ]

        additions = available.sample(
            n=remaining,
            random_state=random_state + 1
        )

        selected_final = pd.concat(
            [selected_final, additions],
            ignore_index=True
        )

    return selected_final

def load_selected_cities(metadata_path):
    selected = pd.read_csv(metadata_path)

    required = ["City", "Country", "Latitude", "Longitude"]

    if selected[required].isna().any().any():
        raise ValueError("Şehir metadata dosyasında eksik kimlik bilgisi var.")

    if len(selected) != 200:
        raise ValueError(f"200 şehir bekleniyordu, {len(selected)} bulundu.")

    if selected[["Latitude", "Longitude"]].duplicated().any():
        raise ValueError("Tekrarlanan koordinatlar bulundu.")

    return selected

def split_temperature_data(df):
    data = df.copy()
    data["dt"] = pd.to_datetime(data["dt"])

    train_df = data[
        data["dt"].between("1900-01-01", "1989-12-01")
    ].copy()

    val_df = data[
        data["dt"].between("1990-01-01", "2000-12-01")
    ].copy()

    test_df = data[
        data["dt"].between("2001-01-01", "2012-12-01")
    ].copy()

    return train_df, val_df, test_df

from pathlib import Path

def save_processed_data(
    train_df,
    val_df,
    test_df,
    monthly_reference,
    output_dir="data/processed"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train.csv": train_df,
        "validation.csv": val_df,
        "test.csv": test_df,
        "monthly_reference.csv": monthly_reference
    }

    for filename, data in datasets.items():
        data.to_csv(output_dir / filename, index=False)

    print(f"{len(datasets)} dosya kaydedildi: {output_dir}")