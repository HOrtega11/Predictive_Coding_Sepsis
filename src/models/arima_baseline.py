import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

from config import ARIMA_MIN_SERIES_LENGTH


class ARIMABaseline:
    """
    Full-history univariate ARIMA baseline for vital-sign forecasting.

    This baseline fits a separate ARIMA model for each vital sign using all
    available historical values up to the current prediction time point.

    Each vital sign is modeled independently, meaning this baseline does not
    capture interactions between variables.

    Parameters
    ----------
    order : tuple
        ARIMA order specified as (p, d, q), where:
        p = autoregressive order,
        d = differencing order,
        q = moving-average order.
    """

    def __init__(self, order):
        """
        Store the ARIMA configuration.

        Parameters
        ----------
        order : tuple
            ARIMA model order, usually written as (p, d, q).
        """

        self.order = order

    def forecast_series(self, series, steps):
        """
        Forecast future values for a single vital-sign time series.

        Parameters
        ----------
        series : np.ndarray
            One-dimensional historical time series for one vital sign.

        steps : int
            Number of future time steps to forecast.

        Returns
        -------
        np.ndarray
            Forecasted values with length equal to `steps`.

        Notes
        -----
        If the series is too short, constant, or ARIMA fitting fails, the method
        falls back to persistence forecasting by repeating the last observed
        value. This prevents failures during evaluation on short ICU stays.
        """

        # If the time series is too short or constant, ARIMA is unreliable.
        # Use persistence forecast instead.
        if len(series) < ARIMA_MIN_SERIES_LENGTH or np.all(series == series[0]):
            return np.repeat(series[-1], steps)

        try:
            # ARIMA fitting can produce many convergence/stationarity warnings.
            # These are suppressed because failed fits are handled by fallback.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(series, order=self.order)
                model_fit = model.fit()

            # Forecast the requested number of steps into the future.
            forecast = model_fit.forecast(steps=steps)
            return forecast

        except Exception:
            # If fitting or forecasting fails, fall back to persistence.
            return np.repeat(series[-1], steps)

    def predict_patient(self, patient_df, vital_columns, horizon):
        """
        Generate ARIMA predictions for one patient's ICU time series.

        Parameters
        ----------
        patient_df : pandas.DataFrame
            DataFrame containing one patient's time-ordered vital-sign records.

        vital_columns : list of str
            Names of the vital-sign columns to forecast.

        horizon : int
            Prediction horizon in hours/time steps.

        Returns
        -------
        tuple of np.ndarray
            preds : np.ndarray
                Predicted vital signs with shape
                (num_predictions, num_vitals).

            targets : np.ndarray
                True future vital signs with shape
                (num_predictions, num_vitals).

            last_inputs : np.ndarray
                Last observed vital signs before prediction with shape
                (num_predictions, num_vitals).

        Notes
        -----
        For each prediction point i:
        - history uses data from time 0 through i
        - prediction target is data[i + horizon]
        - last input is data[i]

        This makes the model a full-history ARIMA baseline rather than a
        fixed-window ARIMA baseline.
        """

        preds = []
        targets = []
        last_inputs = []

        # Extract vital-sign values as a NumPy array.
        data = patient_df[vital_columns].values

        # For each valid prediction point, forecast `horizon` steps ahead.
        for i in range(len(data) - horizon):
            # Use all available past observations up to the current time.
            history = data[:i + 1]

            pred_step = []

            # Fit one independent ARIMA model per vital sign.
            for v in range(len(vital_columns)):
                series = history[:, v]
                forecast = self.forecast_series(series, steps=horizon)

                # Use the final forecasted step as the horizon-specific prediction.
                pred_step.append(forecast[-1])

            preds.append(pred_step)
            targets.append(data[i + horizon])
            last_inputs.append(data[i])

        return (
            np.array(preds),
            np.array(targets),
            np.array(last_inputs),
        )