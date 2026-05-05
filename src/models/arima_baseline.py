
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

from config import ARIMA_MIN_SERIES_LENGTH


class ARIMABaseline:
    def __init__(self, order):
        self.order = order

    def forecast_series(self, series, steps):
        if len(series) < ARIMA_MIN_SERIES_LENGTH or np.all(series == series[0]):
            return np.repeat(series[-1], steps)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(series, order=self.order)
                model_fit = model.fit()

            forecast = model_fit.forecast(steps=steps)
            return forecast

        except Exception:
            return np.repeat(series[-1], steps)

    def predict_patient(self, patient_df, vital_columns, horizon):
        preds = []
        targets = []
        last_inputs = []

        data = patient_df[vital_columns].values

        for i in range(len(data) - horizon):
            history = data[:i + 1]

            pred_step = []
            for v in range(len(vital_columns)):
                series = history[:, v]
                forecast = self.forecast_series(series, steps=horizon)
                pred_step.append(forecast[-1])

            preds.append(pred_step)
            targets.append(data[i + horizon])
            last_inputs.append(data[i])

        return (
            np.array(preds),
            np.array(targets),
            np.array(last_inputs),
        )
    