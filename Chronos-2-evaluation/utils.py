# Some parts adapted from:
# https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import json
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation import evaluation


def data_gen_chronos(
    file_path,
    start_date="2012-05-01",
    day_slot=288,
    header=None,
):
    try:
        data_seq = pd.read_csv(file_path, header=header).values
        print(f"Loaded data shape: {data_seq.shape}")
    except FileNotFoundError:
        print(f"ERROR: input file was not found in {file_path}.")
        return None

    n_timestamps, n_routes = data_seq.shape

    freq_min = int(24 * 60 / day_slot)
    timestamps = pd.date_range(
        start=start_date, periods=n_timestamps, freq=f"{freq_min}min"
    )

    # Columns = 0, 1, 2... (Sensor Indices)
    # Index = Timestamps
    df_wide = pd.DataFrame(data_seq, index=timestamps)
    df_wide.index.name = "timestamp"

    # This transforms the matrix so each sensor becomes a row entry
    df_wide = df_wide.reset_index()
    df_long = df_wide.melt(
        id_vars=["timestamp"], var_name="item_id", value_name="target"
    )

    # Change numeric '0' to string 'sensor_0' to treat it as a categorical identifier
    df_long["item_id"] = "sensor_" + df_long["item_id"].astype(str)

    df_long = df_long.sort_values(by=["item_id", "timestamp"])

    # df_long.to_parquet(output_path, index=False)

    # print(f"Success! Data saved to {output_path}")
    # print(f"Sample:\n{df_long.head()}")

    return df_long


def data_gen_chronos_bike(
    file_path,
    # output_path="bike_chronos.parquet",
    start_date="2012-05-01",
    day_slot=288,
    header=None,
):
    if file_path == "data/bike_flow_in-out.csv":
        col_a = "bike_in"
        col_b = "bike_out"
    else:
        col_a = "bike_pick"
        col_b = "bike_drop"

    df = pd.read_csv(file_path, header=header)
    print(f"Loaded bike data shape: {df.shape}")

    n_timestamps = df.shape[0]
    freq_min = int(24 * 60 / day_slot)
    timestamps = pd.date_range(
        start=start_date, periods=n_timestamps, freq=f"{freq_min}min"
    )

    pick_cols = [c for c in df.columns if col_a in c]
    drop_cols = [c for c in df.columns if col_b in c]

    df_pick = df[pick_cols].copy()
    df_drop = df[drop_cols].copy()
    df_pick["timestamp"] = timestamps
    df_drop["timestamp"] = timestamps

    df_long_pick = df_pick.melt(
        id_vars="timestamp", var_name="item_id", value_name=col_a
    )
    df_long_drop = df_drop.melt(
        id_vars="timestamp", var_name="item_id", value_name=col_b
    )

    df_long_pick["item_id"] = df_long_pick["item_id"].str.replace(f"_{col_a}", "")
    df_long_drop["item_id"] = df_long_drop["item_id"].str.replace(f"_{col_b}", "")

    df_long = pd.merge(df_long_pick, df_long_drop, on=["timestamp", "item_id"])

    df_long = df_long.sort_values(by=["item_id", "timestamp"])

    # df_long.to_parquet(output_path, index=False)

    # print(f"Success! Data saved to {output_path}")
    # print(df_long.head())
    return df_long


def data_gen_chronos_UrbanEV(
    file_path,
    weather_city_file=None,
    # output_path="UrbanEV_chronos.parquet",
    start_date="2012-05-01",
    day_slot=288,
    header=None,
):
    # 1. Load Source Data
    # Expecting shape [Total_Time_Steps, Num_Routes]
    try:
        data_seq = (
            pd.read_csv(file_path, header=header, usecols=lambda x: x != "time")
            .reset_index(drop=True)
            .values
        )
        print(f"Loaded data shape: {data_seq.shape}")
    except FileNotFoundError:
        print(f"ERROR: input file was not found in {file_path}.")
        return None

    n_timestamps, n_routes = data_seq.shape

    freq_min = int(24 * 60 / day_slot)
    timestamps = pd.date_range(
        start=start_date, periods=n_timestamps, freq=f"{freq_min}min"
    )

    # Columns = 0, 1, 2... (Sensor Indices)
    # Index = Timestamps
    df_wide = pd.DataFrame(data_seq, index=timestamps)
    df_wide.index.name = "timestamp"

    df_wide = df_wide.reset_index()
    df_long = df_wide.melt(
        id_vars=["timestamp"], var_name="item_id", value_name="target"
    )

    # Change numeric '0' to string 'sensor_0' to treat it as a categorical identifier
    df_long["item_id"] = "sensor_" + df_long["item_id"].astype(str)

    if weather_city_file:
        try:
            # Load weather data
            df_weather = pd.read_csv(weather_city_file)

            df_weather["timestamp"] = pd.to_datetime(df_weather["time"])

            df_weather = df_weather[["timestamp", "nRAIN", "T"]]

            # Sort both dataframes by timestamp (Requirement for merge_asof)
            df_long = df_long.sort_values("timestamp")
            df_weather = df_weather.sort_values("timestamp")

            # Merge using standard merge since both have 1-hour granularity
            df_long = pd.merge(df_long, df_weather, on="timestamp", how="left")
        except Exception as e:
            raise ValueError(f"WARNING: Could not process weather data. Error: {e}")

    df_long = df_long.sort_values(by=["item_id", "timestamp"])

    # df_long.to_parquet(output_path, index=False)

    # print(f"Success! Data saved to {output_path}")
    print(f"Sample:\n{df_long.head()}")

    return df_long


def load_dataset(cfg):
    if cfg.dataset == "PeMSD7(M)":
        source_file = "data/V_228.csv"
        # output_file = "data/PeMSD7(M)_chronos.parquet"
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
        )

    elif cfg.dataset == "Urban1":
        source_file = "data/Urban1_V_480.csv"
        # output_file = "data/Urban1_chronos.parquet"
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    elif cfg.dataset == "NYC Citi Bike (pick-drop)":
        with h5py.File("data/bike_data.h5", "r") as f:
            pick = f["bike_pick"][:]  # shape (T, N)
            drop = f["bike_drop"][:]  # shape (T, N)

        T, N = pick.shape

        # build column names
        columns = []
        data = []

        for i in range(N):
            columns.append(f"sensor_{i}_bike_pick")
            columns.append(f"sensor_{i}_bike_drop")
            data.append(pick[:, i])
            data.append(drop[:, i])

        # stack to (T, 2N)
        df = pd.DataFrame(np.stack(data, axis=1), columns=columns)

        source_file = "data/bike_flow_pick-drop.csv"
        # output_file = "data/NYC_Citi_Bike_pick-drop_chronos.parquet"
        df.to_csv(source_file, index=False)
        df_long = data_gen_chronos_bike(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    elif cfg.dataset == "PeMSD4":
        data = np.load("data/PEMS04.npz")["data"][:, :, 0]
        np.savetxt("data/PEMS04.csv", data, delimiter=",")
        source_file = "data/PEMS04.csv"
        # output_file = "data/PeMSD4_chronos.parquet"
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
        )

    elif cfg.dataset == "SZ-taxi":
        source_file = "data/sz_speed.csv"
        # output_file = "data/SZ-taxi_chronos.parquet"
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    elif cfg.dataset == "METR-LA":
        filename = "data/metr-la.h5"

        with h5py.File(filename, "r") as f:
            group = f["df"]

            data = group["block0_values"][:]  # shape: (num_cols, num_rows)
            columns = group["axis0"][:].astype(str)
            index = group["axis1"][:].astype(str)

        df = pd.DataFrame(data, columns=columns, index=index)

        source_file = "data/metr-la.csv"
        # output_file = "data/METR-LA_chronos.parquet"
        df.to_csv(source_file, index=False)
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    elif cfg.dataset == "PEMS-BAY":
        filename = "data/pems-bay.h5"
        dataset = h5py.File(filename, "r")
        df = dataset["speed"]

        with h5py.File(filename, "r") as f:
            group = f["speed"]
            data = group["block0_values"][:]  # shape: (num_cols, num_rows)
            columns = group["axis0"][:].astype(str)
            index = group["axis1"][:].astype(str)

        df = pd.DataFrame(data, columns=columns, index=index)

        source_file = "data/pems-bay.csv"
        # output_file = "data/PEMS-BAY_chronos.parquet"
        df.to_csv(source_file, index=False)
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    elif cfg.dataset == "NYC Citi Bike (in-out)":
        with open("data/flow_bike_nyc_irregular.json", "r") as f:
            data = json.load(f)

        inflow = data["inflow"]
        outflow = data["outflow"]

        sensors = sorted(inflow.keys(), key=int)
        T = len(inflow[sensors[0]])

        columns = []
        data_cols = []

        for s in sensors:
            columns.append(f"sensor_{s}_bike_in")
            columns.append(f"sensor_{s}_bike_out")
            data_cols.append(inflow[s])
            data_cols.append(outflow[s])

        # Stack as (T, 2*N)
        df = pd.DataFrame(np.stack(data_cols, axis=1), columns=columns)

        source_file = "data/bike_flow_in-out.csv"
        # output_file = "data/NYC_Citi_Bike_in-out_chronos.parquet"
        df.to_csv(source_file, index=False)
        df_long = data_gen_chronos_bike(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    elif cfg.dataset == "Seattle loop data":
        dense_mat = np.load("data/dense_mat.npy")
        source_file = "data/Seattle_loop_data.csv"
        # output_file = "data/Seattle_loop_data_chronos.parquet"
        np.savetxt(source_file, dense_mat.T, delimiter=",")
        df_long = data_gen_chronos(
            source_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
        )

    elif cfg.dataset == "UrbanEV":
        source_file = "data/occupancy.csv"
        # output_file = "data/UrbanEV_chronos.parquet"
        weather_file = "data/weather_central.csv"
        df_long = data_gen_chronos_UrbanEV(
            source_file,
            weather_city_file=weather_file,
            # output_path=output_file,
            start_date=cfg.start_date,
            day_slot=int(24 * 60 / cfg.freq),
            header=0,
        )

    else:
        raise ValueError(
            f"Dataset {cfg.dataset} not recognized. Please check the dataset name and ensure the corresponding data file is available."
        )

    return df_long


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# Visualization helper function from https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb
def plot_forecast(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    timeseries_id: str,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    history_length: int = 256,
    title_suffix: str = "",
):
    ts_context = context_df.query(f"{id_column} == @timeseries_id").set_index(
        timestamp_column
    )[target_column]
    ts_pred = pred_df.query(
        f"{id_column} == @timeseries_id and target_name == @target_column"
    ).set_index(timestamp_column)[["0.1", "predictions", "0.9"]]
    ts_ground_truth = test_df.query(f"{id_column} == @timeseries_id").set_index(
        timestamp_column
    )[target_column]

    last_date = ts_context.index.max()
    start_idx = max(0, len(ts_context) - history_length)
    plot_cutoff = ts_context.index[start_idx]
    ts_context = ts_context[ts_context.index >= plot_cutoff]
    ts_pred = ts_pred[ts_pred.index >= plot_cutoff]
    ts_ground_truth = ts_ground_truth[ts_ground_truth.index >= plot_cutoff]

    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    ts_context.plot(ax=ax, label=f"historical {target_column}", color="xkcd:azure")
    ts_ground_truth.plot(
        ax=ax, label=f"future {target_column} (ground truth)", color="xkcd:grass green"
    )
    ts_pred["predictions"].plot(ax=ax, label="forecast", color="xkcd:violet")
    ax.fill_between(
        ts_pred.index,
        ts_pred["0.1"],
        ts_pred["0.9"],
        alpha=0.7,
        label="prediction interval",
        color="xkcd:light lavender",
    )
    ax.axvline(x=last_date, color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_title(f"{target_column} forecast for {timeseries_id} {title_suffix}")
    fig.show()


def evaluation_loop(
    pipeline,
    df_long,
    cfg,
    experiment_start_time,
    experiment_end_time,
    pred_delta,
    stride_delta,
    history_length,
    prediction_length,
):
    timestamp_column = cfg.timestamp_column
    horizons = cfg.horizons
    freq = cfg.freq
    id_column = cfg.id_column
    target = cfg.target

    current_step = 1
    current_pred_time = experiment_start_time
    max_horizon_delta = pd.Timedelta(minutes=max(horizons))
    safe_experiment_end = experiment_end_time - max_horizon_delta
    total_steps = int((safe_experiment_end - experiment_start_time) / stride_delta)
    pbar = tqdm(total=total_steps)

    horizon_metrics = {
        h: {
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "COVERAGE": [],
            "IQR_MEAN": [],
            "IQR_MEDIAN": [],
            "IQR_STD": [],
        }
        for h in horizons
    }

    all_preds = {h: [] for h in horizons}
    all_trues = {h: [] for h in horizons}

    # Determine columns for context dataframe
    context_columns = [id_column, timestamp_column, target]
    if getattr(cfg, "past_covariates", None) is not None:
        # Add past covariates, ensuring no duplicates
        for col in cfg.past_covariates:
            if col not in context_columns:
                context_columns.append(col)

    while current_step <= total_steps:
        # print(f"Step {current_step}/{total_steps}: Predicting for time {current_pred_time}")

        # Give context (target + past covariates)
        context_start_time = current_pred_time - pd.Timedelta(
            minutes=history_length * freq
        )
        context_df = df_long[
            (df_long[timestamp_column] >= context_start_time)
            & (df_long[timestamp_column] < current_pred_time)
        ][context_columns].copy()

        # Define the test slice for evaluation
        test_slice_end = current_pred_time + pred_delta
        test_df = df_long[
            (df_long[timestamp_column] >= current_pred_time)
            & (df_long[timestamp_column] < test_slice_end)
        ].copy()

        # Prepare future covariates if needed
        if cfg.future_covariates is not None:
            future_slice_end = current_pred_time + pred_delta
            future_df = df_long[
                (df_long[timestamp_column] >= current_pred_time)
                & (df_long[timestamp_column] < future_slice_end)
            ][[id_column, timestamp_column] + cfg.future_covariates].copy()
        else:
            future_df = None

        # Make forecast
        forecast_df = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=cfg.quantile_levels,
            id_column=cfg.id_column,
            timestamp_column=cfg.timestamp_column,
            target=cfg.target,
            cross_learning=cfg.cross_learning,
        )

        if current_step % 10 == 0:
            pbar.update(10)

        preds = forecast_df[[id_column, timestamp_column, "predictions"]]
        true_vals = test_df[[id_column, timestamp_column, target]]

        # Merge on sensor and timestamp
        merged_df = pd.merge(
            true_vals, preds, on=[id_column, timestamp_column], how="inner"
        )

        if cfg.dataset == "UrbanEV":
            for h in horizons:
                target_ts = current_pred_time + pd.Timedelta(minutes=h)
                step_df = merged_df[merged_df[timestamp_column] == target_ts]
                step_true = step_df[target].values
                step_pred = step_df["predictions"].values

                if len(step_true) > 0:
                    all_trues[h].append(step_true)
                    all_preds[h].append(step_pred)

                    forecast_h = forecast_df[forecast_df[timestamp_column] == target_ts]
                    true_h = test_df[test_df[timestamp_column] == target_ts]
                    prob = probabilistic_metrics(
                        forecast_df=forecast_h,
                        true_df=true_h,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        target_column=target,
                    )

                    horizon_metrics[h]["COVERAGE"].append(prob.get("coverage"))
                    horizon_metrics[h]["IQR_MEAN"].append(prob.get("iqr_mean"))
                    horizon_metrics[h]["IQR_MEDIAN"].append(prob.get("iqr_median"))
                    horizon_metrics[h]["IQR_STD"].append(prob.get("iqr_std"))

        else:
            for h in horizons:
                target_ts = current_pred_time + pd.Timedelta(minutes=h)
                step_df = merged_df[merged_df[timestamp_column] == target_ts]
                step_true = step_df[target].values
                step_pred = step_df["predictions"].values

                if len(step_true) > 0:
                    if cfg.dataset == "METR-LA":
                        step_mape, step_mae, step_rmse = calculate_metrics(
                            pd.DataFrame(step_pred), pd.DataFrame(step_true), null_val=0
                        )
                    else:
                        step_mape, step_mae, step_rmse = evaluation(
                            step_true, step_pred
                        )

                    horizon_metrics[h]["MAE"].append(step_mae)
                    horizon_metrics[h]["MAPE"].append(step_mape)
                    horizon_metrics[h]["RMSE"].append(step_rmse)

                    forecast_h = forecast_df[forecast_df[timestamp_column] == target_ts]
                    true_h = test_df[test_df[timestamp_column] == target_ts]
                    prob = probabilistic_metrics(
                        forecast_df=forecast_h,
                        true_df=true_h,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        target_column=target,
                    )

                    horizon_metrics[h]["COVERAGE"].append(prob.get("coverage"))
                    horizon_metrics[h]["IQR_MEAN"].append(prob.get("iqr_mean"))
                    horizon_metrics[h]["IQR_MEDIAN"].append(prob.get("iqr_median"))
                    horizon_metrics[h]["IQR_STD"].append(prob.get("iqr_std"))
                else:
                    raise ValueError(
                        f"Warning: No data to evaluate for horizon {h} at step {current_step} and {current_pred_time}"
                    )

        current_step += 1
        current_pred_time += stride_delta

    # End of loop - calculate average metrics for each horizon if UrbanEV, otherwise they are already averaged
    if cfg.dataset == "UrbanEV":
        horizon_metrics = {}
        for h in horizons:
            y_true = np.concatenate(all_trues[h])
            y_pred = np.concatenate(all_preds[h])

            mape, mae, rmse = metrics(y_pred, y_true)

            horizon_metrics[h] = {"MAE": mae, "MAPE": mape, "RMSE": rmse}

    pbar.close()

    return horizon_metrics


# Adapted from https://github.com/liyaguang/DCRNN to ensure we have a consistent evaluation metric for METR-LA
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype("float32")
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype("float32")
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype("float32"), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(
        preds=df_pred.values, labels=df_test.values, null_val=null_val
    )
    mae = masked_mae_np(preds=df_pred.values, labels=df_test.values, null_val=null_val)
    rmse = masked_rmse_np(
        preds=df_pred.values, labels=df_test.values, null_val=null_val
    )
    return np.array([mape, mae, rmse])


# Adapted function from https://github.com/IntelligentSystemsLab/UrbanEV to make sure we have a consistent evaluation metric for UrbanEV
def metrics(test_pre, test_real):
    eps = 2e-2
    MAPE_test_real = test_real.copy()
    MAPE_test_pre = test_pre.copy()
    MAPE_test_real[np.where(MAPE_test_real <= eps)] = (
        np.abs(MAPE_test_real[np.where(MAPE_test_real <= eps)]) + eps
    )
    MAPE_test_pre[np.where(MAPE_test_real <= eps)] = (
        np.abs(MAPE_test_pre[np.where(MAPE_test_real <= eps)]) + eps
    )

    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    RAE = np.sum(abs(MAPE_test_pre - MAPE_test_real)) / np.sum(
        abs(np.mean(MAPE_test_real) - MAPE_test_real)
    )

    output_list = np.array([MAPE, MAE, RMSE])
    return output_list


def probabilistic_metrics(
    forecast_df: pd.DataFrame,
    true_df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
    lower_q: float = 0.1,
    upper_q: float = 0.9,
):
    def _find_quantile_col(df, q: float):
        # Try to locate a column whose name parses to the requested quantile
        q_col = None
        for c in df.columns:
            try:
                if abs(float(c) - q) < 1e-8:
                    q_col = c
                    break
            except Exception:
                continue

        if q_col is None:
            # fallback to string form
            s = str(q)
            if s in df.columns:
                q_col = s

        return q_col

    # Locate lower/upper quantile columns (scalar per-row expected)
    lower_col = _find_quantile_col(forecast_df, lower_q)
    upper_col = _find_quantile_col(forecast_df, upper_q)

    if lower_col is None or upper_col is None:
        raise ValueError(
            f"Required quantile columns {lower_q} or {upper_q} not found in forecast_df"
        )

    # Merge forecast and truth on id + timestamp to align rows
    merged = pd.merge(
        true_df[[id_column, timestamp_column, target_column]],
        forecast_df[[id_column, timestamp_column, lower_col, upper_col]],
        on=[id_column, timestamp_column],
        how="inner",
    )

    if merged.shape[0] == 0:
        raise ValueError(
            "No overlapping rows between forecast_df and true_df to evaluate."
        )

    y_true = merged[target_column].values
    lower_vals = merged[lower_col].values.astype(float)
    upper_vals = merged[upper_col].values.astype(float)

    # Compute empirical coverage
    inside = (y_true >= lower_vals) & (y_true <= upper_vals)
    coverage_count = int(np.sum(inside))
    total = int(len(y_true))
    coverage = coverage_count / total

    # Compute IQR per-row using the chosen lower/upper quantiles
    iqr_vals = upper_vals - lower_vals
    iqr_mean = float(np.mean(iqr_vals))
    iqr_median = float(np.median(iqr_vals))
    iqr_std = float(np.std(iqr_vals))

    return {
        "coverage": coverage,
        "coverage_count": coverage_count,
        "total": total,
        "iqr_mean": iqr_mean,
        "iqr_median": iqr_median,
        "iqr_std": iqr_std,
    }
