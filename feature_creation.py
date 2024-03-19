import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def feature_creation_every_row(df):
    df["date"] = pd.to_datetime(df["date"])

    # hour
    # df["hour"] = df.date.dt.hour # change in params.py to match
    df["sin_hour"] = np.sin(2 * np.pi * df.date.dt.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df.date.dt.hour / 24)
    # dayofweek
    # df["dayofweek"] = df.date.dt.dayofweek
    df["sin_day_week"] = np.sin(2 * np.pi * df.date.dt.day_of_week / 7)
    df["cos_day_week"] = np.cos(2 * np.pi * df.date.dt.day_of_week / 7)

    # dayofyear

    # df["dayofyear"] = df.date.dt.dayofyear
    df["sin_day_of_year"] = np.sin(2 * np.pi * df.date.dt.day_of_year / 365)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df.date.dt.day_of_year / 365)

    # week of year
    # df["weekofyear"] = df.date.dt.isocalendar().week
    df["sin_week_year"] = np.sin(2 * np.pi * df.date.dt.isocalendar().week / 52)
    df["cos_week_year"] = np.cos(2 * np.pi * df.date.dt.isocalendar().week / 52)

    # month
    # df["month"] = df.date.dt.month
    df["sin_month"] = np.sin(2 * np.pi * df.date.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df.date.dt.month / 12)

    # quarter
    # df["quarter"] = df.date.dt.quarter
    df["sin_quarter"] = np.sin(2 * np.pi * df.date.dt.quarter / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * df.date.dt.quarter / 4)

    # year
    df["year"] = df.date.dt.year

    return df


import pandas as pd


def feature_creation_group_fuel_type(df, window_sizes):
    for window_size in window_sizes:
        rolling_means = pd.DataFrame(index=df.index)
        rolling_stds = pd.DataFrame(index=df.index)

        for type_name in df["type-name"].unique():
            dft = df[df["type-name"] == type_name]

            rolling_mean = dft["value"].rolling(window=window_size).mean()
            rolling_std = dft["value"].rolling(window=window_size).std()

            rolling_means[type_name] = rolling_mean
            rolling_stds[type_name] = rolling_std

        df[f"rolling_type_mean_{window_size}"] = rolling_means.sum(axis=1)
        df[f"rolling_type_std_{window_size}"] = rolling_stds.sum(axis=1)

    return df


def big_window_sizes_shift(df: pd.DataFrame, big_window_sizes: list[int]) -> pd.DataFrame:
    for ws in big_window_sizes:
        df["shifted_" + str(ws)] = df.groupby("type-name")["value"].shift(ws)

    return df


def feature_creation_group_date(df, window_sizes):
    daily_sum = df.groupby("date")["value"].sum()

    for window_size in window_sizes:
        # Calculate rolling mean and std for the current window size
        rolling_means = daily_sum.rolling(window=window_size, min_periods=1).mean()
        rolling_stds = daily_sum.rolling(window=window_size, min_periods=1).std()

        # Map the rolling stats back to the original DataFrame for the current window size
        df[f"rolling_date_mean_{window_size}"] = df["date"].map(rolling_means)
        df[f"rolling_date_std_{window_size}"] = df["date"].map(rolling_stds)

    return df


def normalize_features(df, columns_to_exclude):
    scaler = StandardScaler()

    features = df.columns.difference(columns_to_exclude)
    df[features] = scaler.fit_transform(df[features])

    return df, scaler


def fc(df: pd.DataFrame, time_period: list[int], big_window_sizes: list[int]) -> pd.DataFrame:
    df = feature_creation_every_row(df)
    df = feature_creation_group_date(df, time_period)
    df = feature_creation_group_fuel_type(df, time_period)
    df = big_window_sizes_shift(df, big_window_sizes)

    df["time_idx"] = df.date.factorize()[0]

    df.drop(
        columns=["type-name", "value-units", "respondent"],
        inplace=True,
    )
    df.bfill(inplace=True)
    return df
