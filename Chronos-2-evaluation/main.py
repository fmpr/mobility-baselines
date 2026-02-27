import os
import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import hydra
from omegaconf import DictConfig, OmegaConf
import utils

@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def main(cfg: DictConfig):
    # Load configs
    dataset_cfg_path = cfg.dataset_cfg
    dataset_cfg = OmegaConf.load(f"conf/{dataset_cfg_path}.yaml")
    cfg = OmegaConf.merge(dataset_cfg, cfg)
    print(OmegaConf.to_yaml(cfg))
    utils.set_seed(cfg.seed)

    # Chronos-2 setup
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Dataset preparation
    df_long = utils.load_dataset(cfg)

    # Forecasting configuration
    prediction_length = (
        cfg.prediction_length + 1
    )  # Number of e.g. 5-minute intervals to predict
    id_column = cfg.id_column  # Column identifying different time series (sensors)
    number_of_sensors = df_long[id_column].nunique()
    freq = cfg.freq  # Dataset granularity in minutes
    day_slot = int(24 * 60 / freq)  # Number of time steps in a day
    stride_steps = (
        cfg.stride_steps
    )  # How many time steps to move forward for each prediction
    MINS_PER_WEEK = 10080
    history_length = (
        int(
            MINS_PER_WEEK / freq
        )  # Provide 7 days of history to the model (10080 minutes in a week)
    )

    if cfg.start_test_day is not None and cfg.n_test_days is not None:
        start_test_day = cfg.start_test_day
        n_test_days = cfg.n_test_days
    elif cfg.proportion_test is not None:
        proportion_test = cfg.proportion_test
        start_test_day = int(
            df_long.shape[0] * (1 - proportion_test) / (day_slot * number_of_sensors)
        )
        n_test_days = int(
            df_long.shape[0] * proportion_test / (day_slot * number_of_sensors)
        )
    else:
        raise ValueError(
            "Either start_test_day and n_test_days or proportion_test must be provided in the config."
        )

    start_date = pd.Timestamp(cfg.start_date)
    experiment_start_time = start_date + pd.Timedelta(days=start_test_day)
    experiment_end_time = experiment_start_time + pd.Timedelta(days=n_test_days)
    pred_delta = pd.Timedelta(
        minutes=prediction_length * freq
    )  # How much into the future we predict
    stride_delta = pd.Timedelta(
        minutes=stride_steps * freq
    )  # How much we move forward at each iteration

    print(
        f"Testing from day {start_test_day} to day {start_test_day + n_test_days} (total {n_test_days} days)"
    )

    # Main sliding window loop
    horizon_metrics = utils.evaluation_loop(
        pipeline,
        df_long,
        cfg,
        experiment_start_time,
        experiment_end_time,
        pred_delta,
        stride_delta,
        history_length,
        prediction_length,
    )

    # Final evaluation for each horizon
    print("Final Average Metrics per Horizon:")

    # Collect per-horizon averages
    overall = {"MAE": [], "MAPE": [], "RMSE": [], "COVERAGE": [], "IQR_MEAN": []}

    for h in cfg.horizons:
        print(f"\nHorizon {h} minutes:")
        avg_mae = np.mean(horizon_metrics[h]["MAE"])
        avg_mape = np.mean(horizon_metrics[h]["MAPE"])
        avg_rmse = np.mean(horizon_metrics[h]["RMSE"])
        avg_coverage = np.mean(horizon_metrics[h]["COVERAGE"])
        avg_iqr = np.mean(horizon_metrics[h]["IQR_MEAN"])

        metrics = {
            "MAE": avg_mae,
            "MAPE": avg_mape,
            "RMSE": avg_rmse,
            "COVERAGE": avg_coverage,
            "IQR_MEAN": avg_iqr,
        }

        # store for overall average
        for k, v in metrics.items():
            overall[k].append(v)

        for name in getattr(cfg, "metrics_to_show", metrics.keys()):
            if name in metrics:
                val = metrics[name]
                if name == "MAPE" or name == "COVERAGE":
                    val *= 100  # Convert to percentage
                    print(f"  {name}: {np.round(val, cfg.decimal_precision)}%")
                else:
                    print(f"  {name}: {np.round(val, cfg.decimal_precision)}")

    print("\nAverage Across Horizons:")
    for name in getattr(cfg, "metrics_to_show", overall.keys()):
        if name in overall and len(overall[name]) > 0:
            val = np.mean(overall[name])
            if name == "MAPE" or name == "COVERAGE":
                val *= 100  # Convert to percentage
                print(f"  {name}: {np.round(val, cfg.decimal_precision)}%")
            else:
                print(f"  {name}: {np.round(val, cfg.decimal_precision)}")


if __name__ == "__main__":
    main()
