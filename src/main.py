import matplotlib.pyplot as plt
from src.dataloader import DataLoader
from src.model import Model
from src.config import *
from src.utils import plot_loss, set_seed
from src.metrics import *
from src.trading_bot import TradingBot
import logging
import argparse

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/combined_config.yml', help='Path to the config file.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--bot', action='store_true', help='Run the trading bot.')
    return parser.parse_args()

def main_train(config):
    loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")

    technical_dataset_train, technical_dataset_valid = loader.prepare_technical_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'],future_steps=config['data']['future_steps'])
    fundamental_dataset_train, fundamental_dataset_valid = loader.prepare_fundamental_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'],future_steps=config['data']['future_steps'])
    combined_dataset_train, combined_dataset_valid = loader.prepare_combined_data(window_size=config['data']['window_size'], overlap=config['data']['overlap'],future_steps=config['data']['future_steps'])


    model_technical = Model(config["model_technical"], config['data'])
    technical_train_loss, technical_valid_loss = model_technical.train(technical_dataset_train, technical_dataset_valid)

    model_fundamental = Model(config["model_fundamental"], config['data'])
    fundamental_train_loss, fundamental_valid_loss = model_fundamental.train(fundamental_dataset_train, fundamental_dataset_valid)

    model_combined = Model(config["model_combined"], config['data'])
    combined_train_loss, combined_valid_loss = model_combined.train(combined_dataset_train, combined_dataset_valid)

    plot_loss(technical_train_loss, technical_valid_loss, "Training and Validation Loss of technical model")
    plot_loss(fundamental_train_loss, fundamental_valid_loss,  "Training and Validation Loss of Fundamental model")
    plot_loss(combined_train_loss, combined_valid_loss,  "Training and Validation Loss of Combined model")
    plt.show()

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
            model = Model(config[m], config['data'])
            model.load_model(f"{config['data']['window_size']}_Best_")
            _, valid_dset = ds_func(config['data']['window_size'], overlap=config['data']['overlap'])

            prediction = model.predict(valid_dset[0])
            evaluate_model(f"{m}", prediction.cpu().numpy(), valid_dset[1], config['data']['threshold'])
            print("============================================================================================")

    if args.bot:
        config = load_config(args.config)

        models = ["model_technical", "model_fundamental", "model_combined"]
        loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv",'data/3_month_thresholds.csv')
        datasets = [loader.prepare_technical_data, loader.prepare_fundamental_data, loader.prepare_combined_data]
        scalers = [loader.technical_scaler, loader.fundamental_scaler, loader.combined_scaler]

        for m, ds_func,scaler in zip(models, datasets, scalers):
            model = Model(config[m], config['data'])
            model.load_model(f"{config['data']['window_size']}_Best_")
            _, valid_dset = ds_func(config['data']['window_size'], overlap=config['data']['overlap'],future_steps=config['data']['future_steps'])
            _, threshold_valid = loader.get_threshold_val()

            bot = TradingBot(config['data']['future_steps'])
            bot.trade_smart(valid_dset[0], model, scaler, config['data']['threshold'])
            bot.stat(f'TRADE SMART: {m}: ')
            bot = TradingBot(config['data']['future_steps'])
            bot.trade(valid_dset[0], model, scaler, config['data']['threshold'])
            bot.stat(f'TRADE SIMPLE: {m}: ')
if __name__ == '__main__':
    main()