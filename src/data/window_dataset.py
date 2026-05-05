
import numpy as np
import torch
from torch.utils.data import Dataset


class VitalWindowDataset(Dataset):
    def __init__(self, dataframe, vital_columns, window_size, prediction_horizon=1):
        self.X = []
        self.y = []

        df = dataframe.sort_values(
            ["subject_id", "hadm_id", "stay_id", "charttime"]
        )

        for _, stay_df in df.groupby(["subject_id", "hadm_id", "stay_id"]):
            values = stay_df[vital_columns].values.astype(np.float32)

            max_start = len(values) - window_size - prediction_horizon + 1

            if max_start <= 0:
                continue

            for i in range(max_start):
                x_window = values[i : i + window_size]
                y_target = values[i + window_size + prediction_horizon - 1]

                self.X.append(x_window)
                self.y.append(y_target)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    