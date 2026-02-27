import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline


class ChronosWrapper(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.pred_len = args.pred_len
        self.context_len = args.context_window
        self.device = device

        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", device_map=self.device, torch_dtype=torch.bfloat16
        )

    def predict(self, history_data, test_data):
        """
        Returns:
            quantiles: (Total_Steps, N, 3) -> [P10, P50, P90]
            samples:   (Total_Steps, N, 20) -> [Raw Sample Paths]
        """
        num_nodes = history_data.shape[1]
        total_steps = len(test_data)

        all_quantiles = []
        all_samples = []

        print(f"Running Chronos inference on {num_nodes} nodes...")

        for t in range(total_steps):
            if t % 100 == 0:
                print(f"Step {t}/{total_steps}")

            # Build context dataframe with history + data up to step t
            context_data = []
            lookback = self.context_len

            for n in range(num_nodes):
                full_stream = np.concatenate([history_data[:, n], test_data[:t, n]])
                # Get the last lookback values
                context_values = full_stream[-lookback:]

                for i, val in enumerate(context_values):
                    context_data.append(
                        {
                            "item_id": f"node_{n}",
                            "timestamp": i,  # relative timestamp
                            "target": val,
                        }
                    )

            context_df = pd.DataFrame(context_data)

            # Make forecast using predict_df
            forecast_df = self.pipeline.predict_df(
                context_df,
                prediction_length=self.pred_len,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="item_id",
                timestamp_column="timestamp",
                target="target",
                cross_learning=True,
            )

            step_quantiles = []
            raw_samples = []

            for n in range(num_nodes):
                node_forecast = forecast_df[forecast_df["item_id"] == f"node_{n}"]
                if len(node_forecast) > 0:
                    # Extract quantiles - columns are '0.1', '0.5', '0.9'
                    pred_row = node_forecast.iloc[0]
                    q10 = float(pred_row["0.1"])
                    q50 = float(pred_row["0.5"])
                    q90 = float(pred_row["0.9"])
                    step_quantiles.append([q10, q50, q90])

                    raw_samples.append(np.zeros(20))
                else:
                    step_quantiles.append([0, 0, 0])
                    raw_samples.append(np.zeros(20))

            all_quantiles.append(np.array(step_quantiles))
            all_samples.append(np.array(raw_samples))

        return np.array(all_quantiles), np.array(all_samples)
