import numpy as np
import torch
from torch.utils.data import Dataset


class VitalWindowDataset(Dataset):
    """
    PyTorch Dataset for creating fixed-length vital-sign forecasting windows.

    Each example consists of:
    - X: a sequence of past vital signs
    - y: the future vital signs after a specified prediction horizon

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Preprocessed dataframe containing patient/stay identifiers, charttime,
        and normalized vital-sign columns.

    vital_columns : list of str
        Names of the vital-sign columns used as model inputs and targets.

    window_size : int
        Number of past time steps included in each input window.

    prediction_horizon : int, optional
        Number of time steps ahead to predict. Default is 1.

    Notes
    -----
    Windows are generated independently within each ICU stay, grouped by:
    subject_id, hadm_id, and stay_id.

    This prevents windows from crossing patient admissions or ICU stays.
    """

    def __init__(self, dataframe, vital_columns, window_size, prediction_horizon=1):
        """
        Construct all valid input/target windows from the dataframe.
        """

        self.X = []
        self.y = []

        # Sort rows so time-series windows are built in chronological order.
        df = dataframe.sort_values(
            ["subject_id", "hadm_id", "stay_id", "charttime"]
        )

        # Generate windows separately for each ICU stay to avoid leakage
        # across different clinical episodes.
        for _, stay_df in df.groupby(["subject_id", "hadm_id", "stay_id"]):
            values = stay_df[vital_columns].values.astype(np.float32)

            # Number of valid starting positions for this stay.
            # Example:
            # window_size = 2, horizon = 1
            # input:  values[i : i + 2]
            # target: values[i + 2]
            max_start = len(values) - window_size - prediction_horizon + 1

            if max_start <= 0:
                continue

            for i in range(max_start):
                # Past window used as model input.
                # Shape: (window_size, num_vitals)
                x_window = values[i : i + window_size]

                # Future target after the prediction horizon.
                # horizon=1 means the immediate next time step after the window.
                y_target = values[i + window_size + prediction_horizon - 1]

                self.X.append(x_window)
                self.y.append(y_target)

        # Convert lists to tensors for PyTorch DataLoader compatibility.
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        """
        Return the number of windowed examples in the dataset.

        Returns
        -------
        int
            Number of examples.
        """

        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieve one input/target pair.

        Parameters
        ----------
        idx : int
            Index of the requested example.

        Returns
        -------
        tuple
            x : torch.Tensor
                Input window with shape (window_size, num_vitals).

            y : torch.Tensor
                Target vital-sign vector with shape (num_vitals,).
        """

        return self.X[idx], self.y[idx]