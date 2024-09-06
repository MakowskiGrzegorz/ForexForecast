import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dataloader import DataLoader
import logging
from metrics import evaluate_model
from src.trading_bot import TradingBot
logging.basicConfig(level=logging.INFO)

class ARIMAXWrapper():
    def __init__(self, train_data, order=(1, 1, 1), steps = 1):
        # Initialize the ARIMA model with exogenous variables
        self.train_endog = train_data[:, 0]
        self.train_exog = train_data[:, 1:]
        self.steps = steps
        self.model = SARIMAX(self.train_endog, exog=self.train_exog, order=order)
        logging.info("Fitting ARIMAX model into train data")
        self.model_fit = self.model.fit(disp=False)
    def predict(self, exog):
        """
        Predict the next time step using the SARIMAX model.

        :param exog: The exogenous variables for the next time step (shape: [1, num_exog_features]).
        :return: Predicted value for the next time step.
        """
        next_forecast = self.model_fit.forecast(steps=self.steps, exog=exog)
        return next_forecast[-1]
    def validate(self, test_data):
        """
        Validate the ARIMAX model using a sliding window approach.

        :param test_data: The validation dataset [num_samples, num_features].
        :param window_size: The size of the sliding window for the past steps.
        :return: A list of one-step-ahead forecasts.
        """
        test_endog = test_data[0]  # feature[0] as endogenous variable
        test_exog = test_data[1]  # the rest as exogenous variables

        # List to store forecasts
        forecasts = []
        history_endog = list(self.train_endog)
        history_exog = list(self.train_exog)
        # Sliding window approach
        for i in range(len(test_endog)):
            # Forecast the next step
            next_forecast = self.model_fit.forecast(steps=window_size, exog=test_exog[i]).predicted_mean
            forecasts.append(next_forecast[0])

            new_endog = test_endog[i][-1]
            new_exog = test_exog[i][-1]
            # Update the model with the new history
            self.model_fit = self.model_fit.apply(endog=[new_endog], exog=[new_exog])

        return np.array(forecasts)

# Assuming DataLoader is a class that loads and prepares your data
loader = DataLoader('data/EURUSD240_technical.csv', 'data/fundamental_data.csv')

# Prepare training and testing data
train_data, test_data = loader.prepare_arimax_data(True) # no sliding window as arimax does

# Initialize the ARIMAX model
arimax = ARIMAXWrapper(train_data, steps=1)
bot = TradingBot(1)
bot.trade_sarimax(test_data, arimax, loader.arimax_scaler, 0.00073)
bot.stat("SARIMAX SIMPLE")
bot.trade_smart_sarimax(test_data, arimax, loader.arimax_scaler, 0.00073)
bot.stat("SARIMAX SMART")