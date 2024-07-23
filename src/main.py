import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sysidentpy.metrics import mean_squared_error
#from sysidentpy.utils.generate_data import get_miso_data

import torch
import torch.nn as nn
#from sysidentpy.neural_network import NARXNN
#from sysidentpy.basis_function._basis_function import Polynomial
#from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
#from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
from src.dataloader import DataLoader
#from sklearn.model_selection import train_test_split
from src.model import Model
from src.config import *
from src.threshold import get_threshold, decision, Decision
from src.utils import plot_loss
from src.metrics import *
import argparse

import logging

logging.basicConfig(level=logging.INFO)

def test_trading_bot(validation_data, model, scaler):
    """
    Testuje bota handlowego, iterując przez dane walidacyjne i przewidując następną wartość.
    
    :param validation_data: Dane walidacyjne do testowania bota.
    :param model: Model do przewidywania następnej wartości ceny.
    :return: Zwraca całkowity zysk z operacji handlowych.
    """
    leaver = 10
    opened_trade = None  # 'long', 'short' lub None
    total_profit = 0.0
    entry_price = 0.0
    num_trades = 0
    profit_trade = 0
    loss_trade = 0
    for data in validation_data:
        data = np.expand_dims(data, 0)
        pred_price = model.predict(data)

        current_price = data[:,-1,0]
        pad_data = np.zeros((current_price.shape[0], 9)) # we need to drop the date column
        pad_data[:,0:1] = current_price[0]
        #pad_data[:,1:10] = data[:, 1:10]
        current_price_true = scaler.inverse_transform(pad_data)[0][0]
        
        prediction = decision(pred_price.item() - current_price, 0.00072)
        if opened_trade is None:
            if prediction.value == Decision.INC.value:
                opened_trade = 'long'
                entry_price = current_price_true
                #logging.info(f"Open long trade at {current_price}")
            elif prediction.value == Decision.DEC.value:
                opened_trade = 'short'
                entry_price = current_price_true
                #logging.info(f"Open short trade at {current_price}")
            # Jeśli 'noact', nie wykonujemy żadnej akcji
        else:
            if opened_trade == 'long' and prediction.value == Decision.DEC.value:
                # Zamknięcie długiej pozycji i obliczenie zysku
                profit = (current_price_true - entry_price) * leaver
                total_profit += profit
                if profit>0:
                    profit_trade += 1
                else:
                    loss_trade += 1
                opened_trade = None
                num_trades += 1
                #logging.info(f"Close long  trade at {current_price_true} \twith profit: {round(profit,5)} \tTotal profit: {round(total_profit,5)} \tNum trades: {num_trades}")
            elif opened_trade == 'short' and prediction.value == Decision.INC.value:
                # Zamknięcie krótkiej pozycji i obliczenie zysku
                profit = (entry_price - current_price_true) * leaver
                total_profit += profit
                if profit>0:
                    profit_trade += 1
                else:
                    loss_trade += 1
                opened_trade = None
                num_trades += 1
                #logging.info(f"Close short trade at {current_price_true} \twith profit: {round(profit,5)} \tTotal profit: {round(total_profit,5)} \tNum trades: {num_trades}")
            # W przeciwnym razie czekamy

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
    parser.add_argument('--config', type=str, default='config/basic_config.yaml', help='Path to the config file.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--bot', action='store_true', help='Run the trading bot.')
    return parser.parse_args()

def main_train(config):
    loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")

    technical_dataset_train, technical_dataset_valid = loader.prepare_technical_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])
    fundamental_dataset_train, fundamental_dataset_valid = loader.prepare_fundamental_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])

    model_technical = Model(config["model_technical"])
    technical_train_loss, technical_valid_loss = model_technical.train(technical_dataset_train, technical_dataset_valid)
    model_technical.save_model()

    

    model_fundamental = Model(config["model_fundamental"])
    fundamental_train_loss, fundamental_valid_loss = model_fundamental.train(fundamental_dataset_train, fundamental_dataset_valid)
    model_fundamental.save_model()

    plot_loss(technical_train_loss, technical_valid_loss, "Training and Validation Loss of technical model")
    plot_loss(fundamental_train_loss, fundamental_valid_loss, "Training and Validation Loss of Fundamental model")
    plt.show()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    #args.test = True
    if args.train:
        main_train(config)
    
    if args.test:
        config = load_config(args.config)
        model_technical = Model(config["model_technical"])
        model_technical.load_model("Epoch_150")
        #model_fundamental = Model(config["model_fundamental"])
        #model_fundamental.load_model()

        loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
        _, technical_dataset_valid = loader.prepare_technical_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])
        _, fundamental_dataset_valid = loader.prepare_fundamental_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])


        technical_prediction = model_technical.predict(technical_dataset_valid[0]).numpy()
        #fundamental_prediction = model_fundamental.predict(fundamental_dataset_valid[0]).numpy()

        assert np.allclose(technical_dataset_valid[1],fundamental_dataset_valid[1]) # we want to make sure that close price are same for both datasets 
        evaluate_model("Technical", technical_prediction, technical_dataset_valid[1], config['data']['threshold'])
        #evaluate_model("Fundamental", fundamental_prediction, fundamental_dataset_valid[1], config['data']['threshold'])


        # combined_decision = get_combined_decision(technical_prediction, fundamental_prediction, config['data']['threshold'])
        # target_diff = np.diff(fundamental_dataset_valid[1], axis=0)
        # target_decision = np.apply_along_axis(lambda x: int(decision(x, config['data']['threshold']).value), 1, target_diff)

        # print("Combined AUC: ", calculate_auc(target_decision, combined_decision))
        # print("Combined F1-score:", calculate_f1_score(*count_classification_results(combined_decision, target_decision)))
        #evaluate_model("Combined", combined_decision, technical_dataset_valid[1], config['data']['threshold'])


    if args.bot:
        config = load_config(args.config)
        model_technical = Model(config["model_technical"])
        model_technical.load_model("Epoch_100")
        #model_fundamental = Model(config["model_fundamental"])
        #model_fundamental.load_model()

        loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
        technical_dataset_train, technical_dataset_valid = loader.prepare_technical_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])
        fundamental_dataset_train, fundamental_dataset_valid = loader.prepare_fundamental_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'])

        #steps  = technical_dataset_train[0].shape[0]
        technical_profit, technical_profit_trade, technical_loss_trade, technical_num_trades = test_trading_bot(technical_dataset_valid[0], model_technical, loader.technical_scaler)
        logging.info(f"TECHNICAL MODEL \tProfit: {technical_profit} \tNum trades: {technical_num_trades} \tProfit trades: {technical_profit_trade} \tLoss trades: {technical_loss_trade}")
        #fundamental_profit, fundamental_num_trades = test_trading_bot(fundamental_dataset_valid[0], model_fundamental, loader.fundamental_scaler)
        #logging.info(f"FUNDAMENTAL MODEL \tProfit: {fundamental_profit} \tNum trades: {fundamental_num_trades}")


        


        # for i in range(steps):
        #     technical_x = technical_dataset_train[0][i]
        #     technical_y = technical_dataset_train[1][i]
        #     fundamental_x = fundamental_dataset_train[0][i]
        #     fundamental_y = fundamental_dataset_train[1][i]

        #     assert technical_y == fundamental_y
        #     print(technical_x.shape)
        #     print(technical_y.shape)
        #     print(fundamental_x.shape)
        #     print(fundamental_y.shape)
            
        #     technical_pred = model_technical.predict(technical_x)
        #     technical_decision = decision(technical_pred, 0.00072) # TODO this threeshold value should be calculated in training step
            
        #     fundamental_pred = model_fundamental.predict(fundamental_x) 
        #     fundamental_decision = decision(fundamental_pred, 0.00072) # TODO this threeshold value should be calculated in training step


            


        # TODO combine models and run on valid data with simple trading bot


    #close_price = loader.technical_data['close']
    #threeshold = get_threshold(close_price)
    # threeshold = 0.00072

    # df = pd.DataFrame()
    # df['close_valid'] = technical_dataset[3].flatten()
    # df = df.iloc[1:]
    # df['diff'] = np.diff(technical_dataset[3], axis=0)
    # df['decision'] = df['diff'].apply(lambda x: decision(x, threeshold))

    # # Definiowanie kolorów dla różnych decyzji
    # colors = df['decision'].map({"INC": "green", "DEC": "red", "NOACT": "blue"})

    # plt.figure(figsize=(14, 7))
    # plt.scatter(df.index, df['close_valid'], c=colors, label='Close Valid')
    # plt.title('Wartości zamknięcia z kolorowym oznaczeniem decyzji')
    # plt.xlabel('Indeks')
    # plt.ylabel('Wartość zamknięcia')
    # plt.legend()
    # plt.show()
    # print(df)

    # get thresshold
    # dla validow trzeba policzyc I/D/NA za pomoca thresholdu
    # nastepnie to samo dla predictu
    # poziej bedzie trzeba napisac nowa funkcje precyzji i sprawdzic jak to jest precyzyjne?




    #model_fundamental = Model(config["model_fundamental"])


    # step 6 train model
    # for epoch in range(num_epochs):
    #     # Convert numpy arrays to torch tensors
    #     inputs = torch.from_numpy(x_train).float()
    #     targets = torch.from_numpy(y_train).float()

    #     # Forward pass
    #     outputs = model(inputs)
    #     loss = criterion(outputs, targets)

    #     # Backward and optimize
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if (epoch+1) % 10 == 0:
    #         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # #step 7 evaluate model
    # model.eval()
    # with torch.no_grad():
    #     inputs = torch.from_numpy(x_valid).float()
    #     targets = torch.from_numpy(y_valid).float()
    #     outputs = model(inputs)
    #     loss = criterion(outputs, targets)
    #     print('Loss: {:.4f}'.format(loss.item()))


    # #step 8 predict future
    # # TODO no test data


    #step 9 visualize
    # import plotly.graph_objects as go

    # import pandas as pd
    # from datetime import datetime

    # df = pd.read_csv("data/EURUSD240.csv", names=['date','open','high','low','close','volume'], parse_dates=['date'], index_col='date', delimiter=",")
    # #df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

    # fig = go.Figure(data=[go.Candlestick(x=df.index,
    #             open=df['open'],
    #             high=df['high'],
    #             low=df['low'],
    #             close=df['close'])])

    # fig.show()

    # Wizualizacja x_valid, y_valid i targets na wykresie
    # plt.figure(figsize=(12,6))
    # #plt.plot(x_valid[0], label='x_valid')
    # plt.plot(outputs, label='prediction')
    # plt.plot(targets, label='targets')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()





# Generate a dataset of a simulated dynamical system
# 
# 
# x_train, x_valid, y_train, y_valid = get_miso_data(
#     n=1000,
#     colored_noise=False,
#     sigma=0.001,
#     train_percentage=80
# )
# print(x_train.shape, y_train.shape)



# class NARX(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin = nn.Linear(4, 10)
#         self.lin2 = nn.Linear(10, 10)
#         self.lin3 = nn.Linear(10, 1)
#         self.tanh = nn.Tanh()

#     def forward(self, xb):
#         z = self.lin(xb)
#         z = self.tanh(z)
#         z = self.lin2(z)
#         z = self.tanh(z)
#         z = self.lin3(z)
#         return z

# basis_function=Polynomial(degree=1)

# narx_net = NARXNN(
#     net=NARX(),
#     ylag=2,
#     xlag=2,
#     basis_function=basis_function,
#     model_type="NARMAX",
#     loss_func='mse_loss',
#     optimizer='Adam',
#     epochs=200,
#     verbose=False,
#     optim_params={'betas': (0.9, 0.999), 'eps': 1e-05} # optional parameters of the optimizer
# )

# # narx_net.fit(X=x_train, y=y_train)
# yhat = narx_net.predict(X=x_valid, y=y_valid)
# plot_results(y=y_valid, yhat=yhat, n=200)
# ee = compute_residues_autocorrelation(y_valid, yhat)
# plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
# x1e = compute_cross_correlation(y_valid, yhat, x_valid)
# plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
