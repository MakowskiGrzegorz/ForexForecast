import matplotlib.pyplot as plt
import numpy as np
from src.threshold import decision, Decision
class TradingModel:
    """
    Abstract base class for models used in the trading bot.
    """
    def predict(self, data):
        raise NotImplementedError("Subclasses should implement this method")

class LSTMModel(TradingModel):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, data):
        data = np.expand_dims(data, 0)
        pred_price = self.model.predict(data)
        current_price = data[:, -1, 0]
        pad_data = np.zeros((current_price.shape[0], data.shape[2]))  # drop the date column
        pad_data[:, 0:1] = current_price[0]
        current_price_true = self.scaler.inverse_transform(pad_data)[0][0]
        return current_price_true, pred_price.item()

class SARIMAXModel(TradingModel):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        current_price = data[-1, 0]  # Assuming feature[0] is the target variable
        pred_price = self.model.validate(data)  # SARIMAX model predicts based on the last window
        return current_price, pred_price

def test_trading_bot(validation_data, model: TradingModel, decision_threshold=0.00072, leaver=10):
    """
    Simulates a trading bot using model predictions on validation data.

    :param validation_data: Validation data for testing the bot (shape: [num_samples, num_features]).
    :param model: Instance of TradingModel (e.g., LSTMModel or SARIMAXModel).
    :param decision_threshold: Threshold for deciding to buy/sell.
    :param leaver: Leverage factor for calculating profit.
    :return: Total profit, number of profitable trades, number of losing trades, total number of trades.
    """
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
    open_trade_types = []

    prices = []
    times = []

    def open_trade(trade_type, current_price, position):
        nonlocal opened_trade, entry_price
        opened_trade = trade_type
        entry_price = current_price
        open_positions.append(position)
        open_trade_prices.append(current_price)
        open_trade_types.append(trade_type)

    def close_trade(current_price, position):
        nonlocal total_profit, num_trades, profit_trade, loss_trade, opened_trade
        profit = 0.0
        if opened_trade == 'long':
            profit = (current_price - entry_price) * leaver
        elif opened_trade == 'short':
            profit = (entry_price - current_price) * leaver

        total_profit += profit
        if profit > 0:
            profit_trade += 1
        else:
            loss_trade += 1

        close_positions.append(position)
        close_trade_prices.append(current_price)
        num_trades += 1
        opened_trade = None

    for i, data in enumerate(validation_data):
        current_price, pred_price = model.predict(data)

        prices.append(current_price)
        times.append(i)

        prediction = decision(pred_price - current_price, decision_threshold)

        if opened_trade is None:
            if prediction.value == Decision.INC.value:
                open_trade('long', current_price, i)
            elif prediction.value == Decision.DEC.value:
                open_trade('short', current_price, i)
        else:
            if (opened_trade == 'long' and prediction.value == Decision.DEC.value) or \
               (opened_trade == 'short' and prediction.value == Decision.INC.value):
                close_trade(current_price, i)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(times, prices, label='Price')

    # Plot long and short trades with different markers/colors
    for j, pos in enumerate(open_positions):
        if open_trade_types[j] == 'long':
            plt.scatter(pos, open_trade_prices[j], color='green', marker='^', label='Open Long' if j == 0 else "", s=100)
        else:  # short
            plt.scatter(pos, open_trade_prices[j], color='blue', marker='v', label='Open Short' if j == 0 else "", s=100)

    plt.scatter(close_positions, close_trade_prices, color='red', marker='x', label='Close Trade', s=100)
    plt.title('Trading Bot - Open/Close Trades')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return total_profit, profit_trade, loss_trade, num_trades
