import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from dataloader import DataLoader
import logging
from metrics import evaluate_model
from src.threshold import decision, Decision
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def test_trading_bot_sarimax(validation_data, model, scaler):
    """
    Test the trading bot using SARIMAX, iterating over validation data and predicting the next value.

    :param validation_data: Validation data for testing the bot (shape: [num_samples, num_features]).
    :param model: SARIMAX model for predicting the next price.
    :return: Total profit and other trading statistics.
    """
    leaver = 10
    opened_trade = None  # 'long', 'short' or None
    total_profit = 0.0
    entry_price = 0.0
    num_trades = 0
    profit_trade = 0
    loss_trade = 0

    open_positions = []
    close_positions = []
    open_trade_prices = []
    close_trade_prices = []
    open_trade_types = []  # 'long' or 'short'

    prices = []
    times = []

    for i in range(len(validation_data) - 1):
        current_data = validation_data[i, 1:]  # Exogenous data at current step
        current_price = validation_data[i, 0]  # Actual price at current step
        
        current_price_true = scaler.inverse_transform([[current_price] + [0] * (validation_data.shape[1] - 1)])[0][0]

        # Predict the next price
        pred_price = model.predict_next(current_data.reshape(1, -1))
        next_price_true = scaler.inverse_transform([[pred_price] + [0] * (validation_data.shape[1] - 1)])[0][0]

        prices.append(current_price_true)
        times.append(i)

        prediction = decision(next_price_true - current_price_true, 0.00072)

        if opened_trade is None:
            if prediction.value == Decision.INC.value:
                opened_trade = 'long'
                entry_price = current_price_true
                open_positions.append(i)
                open_trade_prices.append(current_price_true)
                open_trade_types.append('long')
            elif prediction.value == Decision.DEC.value:
                opened_trade = 'short'
                entry_price = current_price_true
                open_positions.append(i)
                open_trade_prices.append(current_price_true)
                open_trade_types.append('short')
        else:
            if opened_trade == 'long' and prediction.value == Decision.DEC.value:
                profit = (current_price_true - entry_price) * leaver
                total_profit += profit
                if profit > 0:
                    profit_trade += 1
                else:
                    loss_trade += 1
                close_positions.append(i)
                close_trade_prices.append(current_price_true)
                opened_trade = None
                num_trades += 1
            elif opened_trade == 'short' and prediction.value == Decision.INC.value:
                profit = (entry_price - current_price_true) * leaver
                total_profit += profit
                if profit > 0:
                    profit_trade += 1
                else:
                    loss_trade += 1
                close_positions.append(i)
                close_trade_prices.append(current_price_true)
                opened_trade = None
                num_trades += 1

    # Plotting the results (same as in LSTM version)
    plt.figure(figsize=(14, 7))
    plt.plot(times, prices, label='Price')
    
    for j, pos in enumerate(open_positions):
        if open_trade_types[j] == 'long':
            plt.scatter(pos, open_trade_prices[j], color='green', marker='^', label='Open Long' if j == 0 else "", s=100)
        else:
            plt.scatter(pos, open_trade_prices[j], color='blue', marker='v', label='Open Short' if j == 0 else "", s=100)
    
    plt.scatter(close_positions, close_trade_prices, color='red', marker='x', label='Close Trade', s=100)
    plt.title('Trading Bot - Open/Close Trades')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return total_profit, profit_trade, loss_trade, num_trades
class ARIMAXWrapper():
    def __init__(self, train_data, order=(1, 1, 1)):
        # Initialize the ARIMA model with exogenous variables
        self.train_endog = train_data[:, 0]
        self.train_exog = train_data[:, 1:]

        self.model = SARIMAX(self.train_endog, exog=self.train_exog, order=order)
        logging.info("Fitting ARIMAX model into train data")
        self.model_fit = self.model.fit(disp=False)
    def predict_next(self, exog):
        """
        Predict the next time step using the SARIMAX model.

        :param exog: The exogenous variables for the next time step (shape: [1, num_exog_features]).
        :return: Predicted value for the next time step.
        """
        next_forecast = self.model_fit.forecast(steps=1, exog=exog)
        return next_forecast[0]
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
train_data, test_data = loader.prepare_arimax_data(True)
# change that to also have window values.. almost works... let's see tomorrow xD


# Initialize the ARIMAX model
arimax = ARIMAXWrapper(train_data)
arimax_profit, arimax_profit_trade, arimax_loss_trade, arimax_num_trades = test_trading_bot_sarimax(test_data,arimax, loader.arimax_scaler)
logging.info(f"ARIMAX MODEL \tProfit: {arimax_profit} \tNum trades: {arimax_num_trades} \tProfit trades: {arimax_profit_trade} \tLoss trades: {arimax_loss_trade} % of profit trades {arimax_profit_trade/arimax_num_trades * 100}")
# # Validate the model using the sliding window approach
# forecasts = arimax.validate(test_data, window_size=20)

# # Evaluate the model
# evaluate_model("ARIMAX", forecasts, test_data[0], 0.00073)

# #evaluate_model("ARIMAX", forecasts, test_data[0], 0.00073)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(range(test_data[0].shape[0]), test_data[0].reshape(-1), label='Actual')
# plt.plot(range(test_data[0].shape[0]), forecasts, label='Forecast')
# plt.legend()
# plt.show()