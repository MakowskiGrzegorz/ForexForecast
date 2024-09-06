import matplotlib.pyplot as plt
import numpy as np
from src.threshold import decision, Decision
import logging

class TradingBot():
    def __init__(self, future_steps = 1) -> None:
        self.opened_trade = None
        self.total_profit = 0.0
        self.entry_price = 0.0
        self.num_trades = 0
        self.profit_trade = 0
        self.loss_trade = 0
        self.future_steps = future_steps
        self.time_of_trade = []
        ####
    def _denormalize(self, value, scaler, shape):
        pad_data = np.zeros(shape)
        pad_data[:,0:1] = value
        return scaler.inverse_transform(pad_data)[0][0]

    def _get_threshold(self, thresholds, i):
        if thresholds is not None:
            threshold_val = thresholds[i]  if isinstance(thresholds,list) else thresholds
        else:
            threshold_val = 0
        return threshold_val
    def trade_smart(self, data, model, scaler, thresholds=None):
        #for i,data_point in enumerate(data):
        i=0
        while i < len(data) - self.future_steps:
            data_point = np.expand_dims(data[i], 0) # get data for prediction (1,window_size,num_features)
            predicted_price = model.predict(data_point)
            current_price = data_point[:,-1,0] # get last close price
            current_price = self._denormalize(current_price[0], scaler, (1, data_point.shape[2]))
            pred_price_true = self._denormalize(predicted_price[:,-1].item(), scaler, (1, data_point.shape[2]))

            threshold_val = self._get_threshold(thresholds, i)
            prediction = decision(pred_price_true - current_price, threshold_val) # get next decision based on prediction (INC,DEC,NOACT)

            if not self.opened_trade: # we do not have opened trade, so we may want to open one if prediction is other than no act
                if prediction.value == Decision.NOACT.value:
                    # for now will work with only t+1, but we may want to change the loop later to work for t+future_steps
                    i+= self.future_steps
                    continue# go to next data point
                self.entry_price = current_price
                self.opened_trade = 'long' if prediction.value == Decision.INC.value else 'short'
                self.time_of_trade.append(0)
            else: # we want to close the trade only when the price will go other way as we initially predicted (so for long trade we are waiting for the price to go down)
                if self.opened_trade == 'long' and (prediction.value == Decision.DEC.value or self.time_of_trade[self.num_trades] >=500):
                    profit = current_price - self.entry_price
                elif self.opened_trade == 'short' and (prediction.value == Decision.INC.value or self.time_of_trade[self.num_trades] >=500):
                    profit = self.entry_price - current_price
                else:
                    self.time_of_trade[self.num_trades] +=1
                    i+= self.future_steps
                    continue# wait.. do not close trade yet..
                self.total_profit += profit
                self.num_trades += 1
                self.opened_trade = None
                if profit > 0:
                    self.profit_trade +=1
                else:
                    self.loss_trade +=1
            i+= self.future_steps
    def trade(self, data, model, scaler, threesholds = None):
        i=0
        while i < len(data) - self.future_steps:

            data_point = np.expand_dims(data[i], 0) # get data for prediction (1,window_size,num_features)
            predicted_price = model.predict(data_point) # predict next values (self.future_steps)

            current_price = data_point[:,-1,0] # get last close price
            self.entry_price = self._denormalize(current_price[0], scaler, (current_price.shape[0], data_point.shape[2]))
            pred_price_true = self._denormalize(predicted_price[:,-1].item(), scaler, (current_price.shape[0], data_point.shape[2]))

            threshold_val = self._get_threshold(threesholds, i)
            prediction = decision(pred_price_true - self.entry_price, threshold_val) # get next decision based on prediction (INC,DEC,NOACT)

            if prediction.value == Decision.NOACT.value:
                i+=self.future_steps
                continue # no decision, so move to next one

            #calculate the profit/loss from a decision by taking the future value
            true_price = self._denormalize(data[i+self.future_steps,-1,0], scaler, (current_price.shape[0], data_point.shape[2]))
            if prediction.value == Decision.INC.value:
                profit = true_price - self.entry_price
            if prediction.value == Decision.DEC.value:
                profit = self.entry_price - true_price

            self.num_trades+=1
            self.total_profit += profit
            if profit >=0:
                self.profit_trade += 1
            else:
                self.loss_trade += 1
            i+= self.future_steps
    def trade_smart_sarimax(self, validation_data, model, scaler, threesholds = None):
        """
        Test the trading bot using SARIMAX, iterating over validation data and predicting the next value.

        :param validation_data: Validation data for testing the bot (shape: [num_samples, num_features]).
        :param model: SARIMAX model for predicting the next price.
        :return: Total profit and other trading statistics.
        """
        i = self.future_steps
        while i < len(validation_data) - self.future_steps:
            current_data = validation_data[i-self.future_steps:i, 1:]  # Exogenous data at current step
            current_price = validation_data[i, 0]  # Actual price at current step
            
            current_price_true = self._denormalize(current_price, scaler, (1, validation_data.shape[1]))#scaler.inverse_transform([[current_price] + [0] * (validation_data.shape[1] - 1)])[0][0]

            # Predict the next price
            pred_price = model.predict(current_data)
            next_price_true = self._denormalize(pred_price, scaler, (1, validation_data.shape[1]))#scaler.inverse_transform([[pred_price] + [0] * (validation_data.shape[1] - 1)])[0][0]

            prediction = decision(next_price_true - current_price_true, 0.00072)
            if self.opened_trade is None:
                if prediction.value == Decision.INC.value:
                    self.opened_trade = 'long'
                    self.entry_price = current_price_true
                    self.time_of_trade.append(0)
                elif prediction.value == Decision.DEC.value:
                    self.opened_trade = 'short'
                    self.entry_price = current_price_true
                    self.time_of_trade.append(0)
                i+= self.future_steps
            else:
                if self.opened_trade == 'long' and (prediction.value == Decision.DEC.value or self.time_of_trade[self.num_trades] >=200):
                    profit = (current_price_true - self.entry_price)
                    self.total_profit += profit
                    if profit > 0:
                        self.profit_trade += 1
                    else:
                        self.loss_trade += 1
                    self.opened_trade = None
                    self.num_trades += 1
                    i+= self.future_steps
                elif self.opened_trade == 'short' and prediction.value == (Decision.INC.value or self.time_of_trade[self.num_trades] >=200):
                    profit = (self.entry_price - current_price_true)
                    self.total_profit += profit
                    if profit > 0:
                        self.profit_trade += 1
                    else:
                        self.loss_trade += 1
                    self.opened_trade = None
                    self.num_trades += 1
                    i+= self.future_steps
                else:
                    self.time_of_trade[self.num_trades] +=1
                    i+= self.future_steps


    def trade_sarimax(self, data, model, scaler, thresholds = None):
        i=self.future_steps
        while i < len(data) - self.future_steps:
            data_point = data[i-self.future_steps:i, 1:] # all history data till current time
            self.entry_price = self._denormalize(data[i, 0],scaler, (1, data_point.shape[1]+1))
            predicted_price = model.predict(data_point)
            pred_price_true = self._denormalize(predicted_price, scaler, (1, data_point.shape[1]+1))

            threshold_val = self._get_threshold(thresholds, i)
            prediction = decision(pred_price_true - self.entry_price, threshold_val)
            
            if prediction.value == Decision.NOACT.value:
                i+=self.future_steps
                continue # no decision, so move to next one

            #calculate the profit/loss from a decision by taking the future value
            true_price = self._denormalize(data[i+self.future_steps,0], scaler, (1, data_point.shape[1]+1))
            if prediction.value == Decision.INC.value:
                profit = true_price - self.entry_price
            if prediction.value == Decision.DEC.value:
                profit = self.entry_price - true_price

            self.num_trades+=1
            self.total_profit += profit
            if profit >=0:
                self.profit_trade += 1
            else:
                self.loss_trade += 1
            i+= self.future_steps

    def stat(self, name):
        logging.info(f"{name} \tNum trades: {self.num_trades} \tProfit trades: {self.profit_trade} \tLoss trades: {self.loss_trade} \t% of profit trades {round(self.profit_trade/self.num_trades * 100,2)}")
        print(f" & {self.num_trades} & {self.profit_trade} & {self.loss_trade} & {int(self.profit_trade/self.num_trades * 100)}\%")
    def plot(self, name):
        pass



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
