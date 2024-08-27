import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from src.dataloader import DataLoader
from src.model import Model
from src.config import *
from src.threshold import decision, Decision
from src.utils import plot_loss, set_seed
from src.metrics import *
import argparse

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def test_trading_bot(name, validation_data, model, scaler,thresholds=None):
    """
    Testuje bota handlowego, iterując przez dane walidacyjne i przewidując następną wartość.

    :param validation_data: Dane walidacyjne do testowania bota.
    :param model: Model do przewidywania następnej wartości ceny.
    :return: Zwraca całkowity zysk z operacji handlowych oraz rysuje wykres otwierania i zamykania pozycji.
    """
    leaver = 10
    opened_trade = None  # 'long', 'short' lub None
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

    for i, data in enumerate(validation_data):
        data = np.expand_dims(data, 0)
        pred_price = model.predict(data)

        current_price = data[:,-1,0]
        pad_data = np.zeros((current_price.shape[0], data.shape[2]))  # we need to drop the date column
        pad_data[:,0:1] = current_price[0]
        current_price_true = scaler.inverse_transform(pad_data)[0][0]

        pad_data = np.zeros((pred_price.shape[0], data.shape[2]))  # we need to drop the date column
        pad_data[:,0:1] = pred_price.item()
        pred_price_true = scaler.inverse_transform(pad_data)[0][0]
        # Store price and time
        prices.append(current_price_true)
        times.append(i)
        if thresholds is not None:
            try:
                threshold_val = thresholds[i]
            except:
                threshold_val = threshold_val
        else:
            threshold_val = 0
        prediction = decision(pred_price_true - current_price_true, threshold_val)

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
            # Jeśli 'noact', nie wykonujemy żadnej akcji
        else:
            if opened_trade == 'long' and prediction.value == Decision.DEC.value:
                # Zamknięcie długiej pozycji i obliczenie zysku
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
                # Zamknięcie krótkiej pozycji i obliczenie zysku
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
            # W przeciwnym razie czekamy

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(times, prices, label='Price')
    long_label_added = False
    short_label_added = False
    # Plot long and short trades with different markers/colors
    for j, pos in enumerate(open_positions):
        if open_trade_types[j] == 'long':
            if not long_label_added:
                plt.scatter(pos, open_trade_prices[j], color='green', marker='^', label='Open Long', s=100)
                long_label_added = True
            else:
                plt.scatter(pos, open_trade_prices[j], color='green', marker='^', s=100)
        else:  # short
            if not short_label_added:
                plt.scatter(pos, open_trade_prices[j], color='blue', marker='v', label='Open Short', s=100)
                short_label_added = True
            else:
                plt.scatter(pos, open_trade_prices[j], color='blue', marker='v', s=100)

    plt.scatter(close_positions, close_trade_prices, color='red', marker='x', label='Close Trade', s=100)
    plt.title(f'Model {name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    #plt.show()

    return total_profit, profit_trade, loss_trade, num_trades



def get_combined_decision(technical_prediction, fundamental_prediction, threeshold_val):
    
    
    technical_diff = np.diff(technical_prediction, axis=0)
    fundamental_diff = np.diff(fundamental_prediction, axis=0)
    technical_decision = np.apply_along_axis(lambda x: int(decision(x, threeshold_val).value), 1, technical_diff)
    fundamental_decision = np.apply_along_axis(lambda x: int(decision(x, threeshold_val).value), 1, fundamental_diff)
    
    # Analiza decyzji technicznej i fundamentalnej
    combined_decision = np.zeros_like(technical_decision)
    for i in range(len(technical_decision)):
        if technical_decision[i] == Decision.NOACT.value or fundamental_decision[i] == Decision.NOACT.value:
            combined_decision[i] = Decision.NOACT.value
        elif technical_decision[i] == fundamental_decision[i]:
            combined_decision[i] = technical_decision[i]
        else:
            combined_decision[i] = fundamental_decision[i]
    return combined_decision

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/basic_config.yml', help='Path to the config file.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--bot', action='store_true', help='Run the trading bot.')
    return parser.parse_args()

def main_train(config):
    loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")

    technical_dataset_train, technical_dataset_valid = loader.prepare_technical_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])
    fundamental_dataset_train, fundamental_dataset_valid = loader.prepare_fundamental_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])
    combined_dataset_train, combined_dataset_valid = loader.prepare_combined_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])


    model_technical = Model(config["model_technical"])
    technical_train_loss, technical_valid_loss = model_technical.train(technical_dataset_train, technical_dataset_valid)

    model_fundamental = Model(config["model_fundamental"])
    fundamental_train_loss, fundamental_valid_loss = model_fundamental.train(fundamental_dataset_train, fundamental_dataset_valid)

    model_combined = Model(config["model_combined"])
    combined_train_loss, combined_valid_loss = model_combined.train(combined_dataset_train, combined_dataset_valid)

    #plot_loss(technical_train_loss, technical_valid_loss, "Training and Validation Loss of technical model")
    #plot_loss(fundamental_train_loss, fundamental_valid_loss,  "Training and Validation Loss of Fundamental model")
    #plot_loss(combined_train_loss, combined_valid_loss,  "Training and Validation Loss of Combined model")
    #plt.show()

def main():
    set_seed(2137)
    args = parse_arguments()
    config = load_config(args.config)

    if args.train:
        main_train(config)

    if args.test:
        config = load_config(args.config)
        models = ["model_technical", "model_fundamental", "model_combined"]
        loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
        datasets = [loader.prepare_technical_data, loader.prepare_fundamental_data, loader.prepare_combined_data]
        scalers = [loader.technical_scaler, loader.fundamental_scaler, loader.combined_scaler]

        for m, ds_func,scaler in zip(models, datasets, scalers):
            model = Model(config[m])
            model.load_model(f"{config['data']['window_size']}_Best_")
            _, valid_dset = ds_func(config['data']['window_size'], overlap=config['data']['overlap'])

            prediction = model.predict(valid_dset[0])
            #evaluate_model(f"{m}", prediction.cpu().numpy(), valid_dset[1], config['data']['threshold'])
            #print("============================================================================================")

    if args.bot:
        config = load_config(args.config)

        models = ["model_technical", "model_fundamental", "model_combined"]
        loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv",'data/3_month_thresholds.csv')
        datasets = [loader.prepare_technical_data, loader.prepare_fundamental_data, loader.prepare_combined_data]
        scalers = [loader.technical_scaler, loader.fundamental_scaler, loader.combined_scaler]

        for m, ds_func,scaler in zip(models, datasets, scalers):
            model = Model(config[m])
            model.load_model(f"{config['data']['window_size']}_Best_")
            _, valid_dset = ds_func(config['data']['window_size'], overlap=config['data']['overlap'])
            _, threshold_valid = loader.get_threshold_val()
            profit, profit_trade, loss_trade, num_trades = test_trading_bot(m, valid_dset[0], model, scaler, None)
            logging.info(f"{m} \tNum trades: {num_trades} \tProfit trades: {profit_trade} \tLoss trades: {loss_trade} \t% of profit trades {round(profit_trade/num_trades * 100,2)}")
if __name__ == '__main__':
    main()