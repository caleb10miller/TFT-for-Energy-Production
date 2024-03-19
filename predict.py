import numpy as np
import sqlite3
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tft import MyTemporalFusionTransformer
from config import get_config
from feature_creation import fc, feature_creation_every_row
from model import build_time_series_ds
import matplotlib.pyplot as plt


def make_predictions(config, PRED_START_DATE, model_file_path, prediction_quantile):

    # model and config params
    tft = MyTemporalFusionTransformer.load_from_checkpoint(model_file_path)

    pred_len = config["max_pred_len"]
    encoder_len = config["max_encoder_len"]
    embedding_labels = {
        v: k for k, v in tft.hparams.embedding_labels["fueltype"].items()
    }

    PRED_START_DATE = pd.to_datetime(PRED_START_DATE)
    PRED_END_DATE = PRED_START_DATE + pd.Timedelta(hours=pred_len)

    # time series used for training for config
    _, _, ds = build_time_series_ds(config)

    dsp = ds.get_parameters()
    dsp["predict_mode"] = True

    # db conn and setup
    conn = sqlite3.connect("data/power_production.db")
    df = pd.read_sql("SELECT * FROM power_production", conn).drop_duplicates()
    conn.close()

    df.rename(columns={"period": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date", inplace=True)

    df = fc(df, config["time_windows"], config["large_time_windows"])

    df = df[df.date < PRED_START_DATE].loc[
        lambda x: x.time_idx > x.time_idx.max() - encoder_len
    ]

    encoder_data = df[
        lambda x: x.time_idx > x.time_idx.max() - config["max_encoder_len"]
    ]

    # template for what is copied forward
    last_data = df[lambda x: x.time_idx == x.time_idx.max()]

    # create a df with pred_len time steps with all of the time varying known inputs
    week_preds = (
        pd.date_range(
            start=last_data.date.values[0] + pd.Timedelta(hours=1),
            end=last_data.date.values[0] + pd.Timedelta(hours=168),
            freq="h",
        )
        .to_series()
        .reset_index(drop=True)
        .to_frame(name="date")
        .assign(key=1)
        .merge(
            pd.DataFrame({"fueltype": last_data.fueltype.unique().tolist()}).assign(
                key=1
            ),
            on="key",
        )
        .drop("key", axis=1)
    )
    week_preds = feature_creation_every_row(week_preds)
    week_preds["time_idx"] = (
        week_preds.date.factorize()[0] + encoder_data.time_idx.max() + 1
    )

    df["date"] = pd.to_datetime(df["date"])
    week_preds["date"] = pd.to_datetime(week_preds["date"])

    # fill in all of the rolling columns as possible. when not possible, such as 2 hour rolling average for non-known
    # time step, fill in that spot with data from that same time stamp from 1 year prior

    #### CHATGPT CODE
    # Create a composite key in df
    df["composite_key"] = df["date"].astype(str) + "_" + df["fueltype"]

    # Initialize a dictionary to hold mappings for each x in big_window_sizes
    value_mappings = {}

    for x in config["large_time_windows"]:
        # Compute target dates for initial and fallback searches
        week_preds[f"target_date_{x}"] = week_preds["date"] - pd.Timedelta(hours=x)
        week_preds[f"target_date_1_year_{x}"] = week_preds[
            f"target_date_{x}"
        ] - pd.DateOffset(years=1)

        # Create composite keys for initial and fallback target dates in week_preds
        week_preds[f"composite_key_{x}"] = (
            week_preds[f"target_date_{x}"].astype(str) + "_" + week_preds["fueltype"]
        )
        week_preds[f"composite_key_1_year_{x}"] = (
            week_preds[f"target_date_1_year_{x}"].astype(str)
            + "_"
            + week_preds["fueltype"]
        )

        # Create mappings from df's composite keys to values
        value_mappings[x] = df.set_index("composite_key")["value"]

    # Use the mappings to find values for each x, prioritizing initial target dates over fallback dates
    for x in config["large_time_windows"]:
        week_preds[f"shifted_{x}"] = week_preds[f"composite_key_{x}"].map(
            value_mappings[x]
        )
        week_preds[f"shifted_1_year_{x}"] = week_preds[f"composite_key_1_year_{x}"].map(
            value_mappings[x]
        )

        # Combine initial and fallback values, prioritizing initial where available
        week_preds[f"shifted_{x}"] = week_preds[f"shifted_{x}"].fillna(
            week_preds[f"shifted_1_year_{x}"]
        )

    # Cleanup: drop intermediate columns
    columns_to_drop = [
        col
        for col in week_preds.columns
        if "target_date" in col or "composite_key" in col or "shifted_1_year" in col
    ]
    week_preds.drop(columns=columns_to_drop, inplace=True)
    #### CHATGPT CODE

    new_prediction_data = pd.concat(
        [encoder_data, week_preds], ignore_index=True
    ).fillna(0)

    new_prediction_data = TimeSeriesDataSet.from_parameters(dsp, new_prediction_data)

    # make predictions
    raw_preds = tft.predict(
        new_prediction_data, mode="raw", return_x=True, num_workers=32
    )

    all_raw_preds = raw_preds.output[0].cpu().numpy()

    # predictions into dataframe
    df_preds = pd.DataFrame()

    for x in range(8):
        values = (
            raw_preds.output[0][x][:, prediction_quantile].cpu().numpy()
        )  # [:,3] is 0.5 quantile prediction, adjust for other quantiles quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

        df_preds[x] = values

    df_preds.rename(columns=embedding_labels, inplace=True)
    df_preds.index = pd.date_range(PRED_START_DATE, periods=len(df_preds), freq="h")

    ## true data
    conn = sqlite3.connect("data/power_production.db")
    df_true = pd.read_sql(
        f"SELECT * FROM power_production WHERE date >= '{str(df_preds.index.min())}' AND date <= '{str(df_preds.index.max())}'",
        conn,
    ).drop_duplicates()
    conn.close()

    df_true = df_true.pivot_table(
        index=["date"], columns="fueltype", values="value", aggfunc="sum"
    ).reset_index()

    df_true["date"] = pd.to_datetime(df_true["date"])
    df_true = df_true.drop(columns=["date"])

    df_true.index = pd.date_range(PRED_START_DATE, periods=len(df_true), freq="h")

    # Assuming df_true and df_preds are your DataFrames

    true_values = df_true.values.flatten()
    predicted_values = df_preds.values.flatten()

    # Calculate MAE, MSE, RMSE, and R-squared
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    # adjusted r2
    n = len(true_values)
    p = len(df_preds.columns)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    true_sum = df_true.sum(axis=1)
    pred_sum = df_preds.sum(axis=1)
    # mae rsme and r2 for total power production
    sum_mae = mean_absolute_error(true_sum, pred_sum)
    sum_mse = mean_squared_error(true_sum, pred_sum)
    sum_rmse = np.sqrt(sum_mse)
    # sum_r2 = r2_score(true_sum, pred_sum)

    fig1, axes1 = plt.subplots(4, 2, figsize=(15, 10))
    axes1 = axes1.flatten()

    for i, col in enumerate(df_preds.columns):
        axes1[i].plot(df_preds[col], label="Predicted", color="blue")
        axes1[i].plot(df_true[col], label="True", color="orange")
        axes1[i].set_title(col)
        axes1[i].legend()
        axes1[i].xaxis.set_tick_params(rotation=45)

    formatted_start_date = PRED_START_DATE.strftime("%Y-%m-%d")
    formatted_end_date = PRED_END_DATE.strftime("%Y-%m-%d")
    title1 = f"Hourly Predictions by Fuel Type: {formatted_start_date} to {formatted_end_date} \n MAE: {mae:.2f}, RMSE: {rmse:.2f}"

    fig1.suptitle(title1, fontsize=16)
    fig1.tight_layout()
    plt.close(fig1)  # Prevents auto-display in Jupyter Notebooks

    # Second Figure: Sum Power Production
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(true_sum, label="True", color="orange")
    ax2.plot(pred_sum, label="Predicted", color="blue")
    ax2.set_title(f"Sum Power Production\n MAE: {sum_mae:.2f}, RMSE: {sum_rmse:.2f}")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()

    fig2.tight_layout()
    plt.close(fig2)  # Prevents auto-display in Jupyter Notebooks

    return ((all_raw_preds, df_true), (mae, sum_mae, adj_r2), fig1, fig2, last_data)
