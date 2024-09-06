from src.lstm import LSTMModel
from src.dataloader import DataLoader
from src.config import load_config
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os
from src.model import Model
from src.metrics import calculate_auc
from src.threshold import decision
from src.utils import set_seed
__DEVICE__ = 'cuda' if torch.cuda.is_available() else 'cpu'
def build_model(params,type):
    #model_base = "config/basic_config.yml"
    try:
        model_base = sys.argv[1]
        logging.info(f"Building {model_base}")
    except:
        logging.fatal("No base model selected! Please add model name from configs as first argument")
        exit(-1)
    if not os.path.exists(model_base):
        logging.fatal(f"Config {model_base} does not exist")
        exit(-1)

    cfg = load_config(model_base)
    cfg[f'model_{type}']["hidden_dim"] = params["hidden_size"]
    cfg[f'model_{type}']["layer_dim"] = params["num_layers"]
    cfg[f'model_{type}']["dropout"] = 0.0#params["dropout"]
    cfg[f'model_{type}']["training"]["learning_rate"] = params["learning_rate"]
    cfg[f'model_{type}']["training"]["num_epochs"] = params["epochs"]
    cfg['data']['future_steps'] = params['forward_step']
    model = Model(cfg[f'model_{type}'],cfg['data'])

    return model

def technical_objective(params):
    model = build_model(params, 'technical')
    loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
    technical_dataset_train, technical_dataset_valid = loader.prepare_technical_data(window_size=params['window_size'], overlap=params['overlap'],future_steps=params['forward_step'])

    model.train(technical_dataset_train, technical_dataset_valid)
    test_loss = model.validate(technical_dataset_valid)
    #print(f"Tested:{params} with loss: {test_loss}")
    return {'loss': test_loss, 'status': STATUS_OK}

def fundamental_objective(params):
    model = build_model(params, 'fundamental')
    loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
    fundamental_dataset_train, fundamental_dataset_valid = loader.prepare_fundamental_data(window_size=params['window_size'], overlap=params['overlap'],future_steps=params['forward_step'])

    model.train(fundamental_dataset_train, fundamental_dataset_valid)
    test_loss = model.validate(fundamental_dataset_valid)

    #print(f"Tested:{params} with loss: {test_loss}")
    return {'loss': test_loss, 'status': STATUS_OK}

def combined_objective(params):
    model = build_model(params, 'combined')
    loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
    combined_dataset_train, combined_dataset_valid = loader.prepare_combined_data(window_size=params['window_size'], overlap=params['overlap'],future_steps=params['forward_step'])

    model.train(combined_dataset_train, combined_dataset_valid)
    test_loss = model.validate(combined_dataset_valid)

    #print(f"Tested:{params} with loss: {test_loss}")
    return {'loss': test_loss, 'status': STATUS_OK}


if __name__ == "__main__":
    set_seed(2137)

    hidden_size_space = [25,32, 50,64, 100,128, 150,200,256]
    num_layers_space = [2,3,4]
    epochs_space = [300]
    window_size_space = [10]
    forward_step_space = [5]
    space = {
        'hidden_size': hp.choice('hidden_size', hidden_size_space),
        'num_layers': hp.choice('num_layers', num_layers_space),
        #'dropout': hp.uniform('dropout', 0.0, 0.0),
        'learning_rate': hp.uniform('learning_rate', 1e-5, 1),
        'epochs': hp.choice('epochs', epochs_space),
        'window_size': hp.choice('window_size',window_size_space),
        'overlap': hp.choice('overlap',[True]),
        'forward_step':hp.choice('forward_step',forward_step_space),
    }
    functions = [technical_objective, fundamental_objective, combined_objective]
    for i, type in enumerate(["technical", "fundamental", 'combined']):

        trials = Trials()
        best_model = fmin(fn=functions[i],
                          space=space,
                          algo=tpe.suggest,
                          max_evals=80,
                          trials=trials)
        best_params = {
            'hidden_size': hidden_size_space[best_model['hidden_size']],
            'num_layers': num_layers_space[best_model['num_layers']],
            'dropout': 0.0,#best_model['dropout'],
            'learning_rate': best_model['learning_rate'],
            'epochs': epochs_space[best_model['epochs']],
            'window_size': window_size_space[best_model['window_size']],
        }
        print(f"Best {type} parameters:", best_params)