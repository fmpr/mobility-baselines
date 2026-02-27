# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import chronos_model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class CreateDataset(Dataset):
    def __init__(self, args, occ, extra_feat, device):  # adj
        lb = args.seq_len
        pt = args.pred_len
        self.pred_type = args.pred_type
        occ, label = create_rnn_data(occ, lb, pt)
        self.occ = torch.Tensor(occ)
        self.label = torch.Tensor(label)

        self.extra_feat = "None"
        if extra_feat != "None":
            extra_feat, _ = create_rnn_data(extra_feat, lb, pt)
            self.extra_feat = torch.Tensor(extra_feat)
        self.device = device

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):  # occ: batch, seq, node
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        if self.extra_feat != "None":
            output_extra_feat = torch.transpose(self.extra_feat[idx, :, :], 0, 1).to(
                self.device
            )
            return output_occ, output_label, output_extra_feat
        else:
            return output_occ, output_label


def create_loaders(
    train_occ,
    valid_occ,
    test_occ,
    train_extra_feat,
    valid_extra_feat,
    test_extra_feat,
    args,
    device,
):
    train_dataset = CreateDataset(args, train_occ, train_extra_feat, device)
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, drop_last=True
    )

    valid_dataset = CreateDataset(args, valid_occ, valid_extra_feat, device)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occ), shuffle=False)

    test_dataset = CreateDataset(args, test_occ, test_extra_feat, device)
    test_loader = DataLoader(test_dataset, batch_size=len(test_occ), shuffle=False)

    return train_loader, valid_loader, test_loader


def read_data(args):
    """
    Read and preprocess the dataset for model input.
    """

    # Load datasets
    inf = pd.read_csv("../data/inf.csv", header=0, index_col=None)
    occ = pd.read_csv("../data/occupancy.csv", header=0, index_col=0)
    duration = pd.read_csv("../data/duration.csv", header=0, index_col=0)
    volume = pd.read_csv("../data/volume.csv", header=0, index_col=0)
    e_price = pd.read_csv("../data/e_price.csv", index_col=0, header=0).values
    s_price = pd.read_csv("../data/s_price.csv", index_col=0, header=0).values
    adj = pd.read_csv("../data/adj.csv", header=0, index_col=None)
    adj.index = adj.columns

    time = pd.to_datetime(occ.index)

    feat = occ
    if args.feat == "duration":
        feat = duration
    elif args.feat == "volume":
        feat = volume

    # Normalize
    charge_count_dict = dict(zip(inf["TAZID"].astype(str), inf["charge_count"]))
    for col in occ.columns:
        charge_count = charge_count_dict[col]
        occ[col] = occ[col] / charge_count

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    e_price = price_scaler.fit_transform(e_price)
    s_price = price_scaler.fit_transform(s_price)

    # Load weather data
    weather = pd.read_csv(r"../data/weather_central.csv", header=0, index_col="time")

    extra_feat = "None"
    if args.add_feat != "None":
        extra_feat = np.zeros([occ.shape[0], occ.shape[1], 1])
        add_feat_list = args.add_feat.split("+")
        for add_feat in add_feat_list:
            if add_feat == "all":
                extra_feat = np.concatenate(
                    [extra_feat, e_price[:, :, np.newaxis]], axis=2
                )
                extra_feat = np.concatenate(
                    [extra_feat, s_price[:, :, np.newaxis]], axis=2
                )
                extra_feat = np.concatenate(
                    [
                        extra_feat,
                        np.repeat(
                            weather.values[:, np.newaxis, :], occ.shape[1], axis=1
                        ),
                    ],
                    axis=2,
                )
            elif add_feat == "e":
                extra_feat = np.concatenate(
                    [extra_feat, e_price[:, :, np.newaxis]], axis=2
                )
            elif add_feat == "s":
                extra_feat = np.concatenate(
                    [extra_feat, s_price[:, :, np.newaxis]], axis=2
                )
            else:
                extra_feat = np.concatenate(
                    [
                        extra_feat,
                        np.repeat(
                            weather[add_feat].values[:, np.newaxis, np.newaxis],
                            occ.shape[1],
                            axis=1,
                        ),
                    ],
                    axis=2,
                )
        extra_feat = extra_feat[:, :, 1:]

    return np.array(feat), np.array(adj), extra_feat, time


def set_seed(seed, flag):
    if flag:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def division(data, train_rate=0.7, valid_rate=0.2, test_rate=0.1):
    data_length = len(data)
    train_division_index = int(data_length * train_rate)
    valid_division_index = int(data_length * (train_rate + valid_rate))
    test_division_index = int(data_length * (1 - test_rate))
    train_data = data[:train_division_index]
    valid_data = data[train_division_index:valid_division_index]
    test_data = data[test_division_index:]
    return train_data, valid_data, test_data


def load_net(args, adj, device, occ):
    adj_dense = torch.Tensor(adj).to(device)
    num_node = occ.shape[1] if args.pred_type == "region" else 1
    n_fea = 1
    if args.add_feat == "all":
        n_fea = 8
    elif args.add_feat == "None":
        n_fea = 1
    else:
        for _ in args.add_feat.split("+"):
            n_fea += 1
    if args.model == "chronos-2":
        model = chronos_model.ChronosWrapper(args, device)
    return model


def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i : i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)


def metrics(test_pre, test_real, args):
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

    #print("MAPE: {}".format(MAPE))
    #print("MAE:{}".format(MAE))
    #print("MSE:{}".format(MSE))
    #print("RMSE:{}".format(RMSE))
    #print("RAE:{}".format(RAE))
    output_list = [MSE, RMSE, MAPE, RAE, MAE]
    return output_list


def split_cv(
    args,
    time,
    feat,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    extra_feat="None",
):
    """
    Split dataset based on time for time-series rolling cross-validation.
    """
    assert len(time) == len(feat)
    fold = args.fold
    month_list = list(time.month.unique())
    assert args.total_fold == len(month_list)
    fold_time = time.month.isin(month_list[0:fold]).sum()

    train_end = int(fold_time * train_ratio)
    valid_start = train_end
    valid_end = int(valid_start + fold_time * valid_ratio)
    test_start = valid_end
    test_end = int(fold_time)

    train_feat = feat[:train_end]
    valid_feat = feat[valid_start:valid_end]
    test_feat = feat[test_start:test_end]

    scaler = "None"

    if args.pred_type == "region":
        if args.feat != "occ":
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat)
            valid_feat = scaler.transform(valid_feat)
            test_feat = scaler.transform(test_feat)
    else:
        node_idx = int(args.pred_type)
        if args.feat != "occ":
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat[:, node_idx].reshape(-1, 1))
            valid_feat = scaler.transform(valid_feat[:, node_idx].reshape(-1, 1))
            test_feat = scaler.transform(test_feat[:, node_idx].reshape(-1, 1))
        else:
            train_feat = train_feat[:, node_idx].reshape(-1, 1)
            valid_feat = valid_feat[:, node_idx].reshape(-1, 1)
            test_feat = test_feat[:, node_idx].reshape(-1, 1)

    train_extra_feat, valid_extra_feat, test_extra_feat = "None", "None", "None"
    if extra_feat != "None":
        train_extra_feat = extra_feat[:train_end]
        valid_extra_feat = extra_feat[valid_start:valid_end]
        test_extra_feat = extra_feat[test_start:test_end]

    return (
        train_feat,
        valid_feat,
        test_feat,
        train_extra_feat,
        valid_extra_feat,
        test_extra_feat,
        scaler,
    )


def probabilistic_metrics(
    forecast_df: pd.DataFrame,
    true_df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
    lower_q: float = 0.1,
    upper_q: float = 0.9,
):
    """
    Compute empirical coverage and IQR metrics for probabilistic forecasts.
    
    Args:
        forecast_df: DataFrame with forecast quantiles
        true_df: DataFrame with true values
        id_column: Column name for identifiers
        timestamp_column: Column name for timestamps
        target_column: Column name for target variable
        lower_q: Lower quantile level (default 0.1)
        upper_q: Upper quantile level (default 0.9)
    
    Returns:
        dict: Dictionary containing coverage, iqr_mean, iqr_median, iqr_std
    """
    def _find_quantile_col(df, q: float):
        # Locate a column whose name parses to the requested quantile
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

    # Locate lower/upper quantile columns
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


def compute_metrics(csv_path):
    """
    Reads a CSV of model metrics and computes mean MAE, MAPE, RMSE, RAE, Coverage, and IQR per prediction horizon.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Table with metrics rounded to 2 decimals, indexed by pred_len.
    """
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)

    desired_metrics = ["MAE", "MAPE", "RMSE", "RAE", "COVERAGE", "IQR_MEAN"]

    def find_col_ci(df_cols, target):
        for c in df_cols:
            if c.strip().lower() == target.strip().lower():
                return c
        return None

    pred_col = find_col_ci(df.columns, "pred_len")
    if pred_col is None:
        raise KeyError("Required column 'pred_len' not found in CSV")

    metrics_cols = []
    for m in desired_metrics:
        found = find_col_ci(df.columns, m)
        if found is None:
            df[m] = np.nan
        else:
            if found != m:
                df[m] = df[found]
        metrics_cols.append(m)

    # Group by prediction horizon and compute mean (multiply by 100 for percentage-like metrics)
    summary = df.groupby(pred_col)[metrics_cols].mean()
    summary = summary.round(2)
    print(summary)
