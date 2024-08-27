import torch
from torch import nn
from src.lstm import LSTMModel
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.threshold import decision
import logging

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

__DEVICE__ = 'cuda' if torch.cuda.is_available() else 'cpu'
class DirectionalMeanSquaredError(nn.Module): # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3560007
    """
    Klasa implementująca niestandardową funkcję straty, która minimalizuje różnicę
    między przewidywaną a rzeczywistą wartością oraz dopasowuje różnicę do różnicy celu.
    """
    def __init__(self):
        super(DirectionalMeanSquaredError, self).__init__()

    def forward(self, predictions, targets):
        """
        Oblicza niestandardową stratę MSE uwzględniającą kierunek zmiany ceny.

        :param predictions: Przewidywane wartości przez model.
        :param targets: Rzeczywiste wartości.
        :param target_diff: Rzeczywista różnica między kolejnymi wartościami w danych.
        :return: Wartość straty.
        """

        pred_1 = torch.roll(predictions, 1)
        target_1 = torch.roll(targets, 1)

        sign = torch.sign((targets - target_1) * (predictions  - pred_1))

        return torch.mean(torch.pow((predictions - targets) - 0.01*sign, 2))

class SimpleDirectionalMeanSquaredError(nn.Module):
    def __init__(self):
        super(SimpleDirectionalMeanSquaredError, self).__init__()

    def forward(self, predictions, targets):
        #Oblicz różnicę między przewidywaniami a rzeczywistymi wartościami
        pred_diff = predictions[1:] - predictions[:-1]
        target_diff = targets[1:] - targets[:-1]
        #Oblicz błąd kwadratowy między przewidywaną a rzeczywistą różnicą
        loss = torch.mean((pred_diff - target_diff) ** 2)
        return loss * 1000
class CombinedLoss(nn.Module):
    def __init__(self, weight_mse=0.6, weight_directional=0.4):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.directional_loss = SimpleDirectionalMeanSquaredError()
        self.weight_mse = weight_mse
        self.weight_directional = weight_directional

    def forward(self, predictions, targets):
        """
        Oblicza połączoną stratę, która jest ważoną sumą MSE i niestandardowej straty kierunkowej.

        :param predictions: Przewidywane wartości przez model.
        :param targets: Rzeczywiste wartości.
        :return: Wartość połączonej straty.
        """
        # Standardowa strata MSE
        loss_mse = self.mse_loss(predictions, targets)
        # Niestandardowa strata kierunkowa
        loss_directional = self.directional_loss(predictions, targets)
        # Połączona strata z ważoną sumą
        combined_loss = (self.weight_mse * loss_mse) + (self.weight_directional * loss_directional)
        return combined_loss


class BestModelSaver:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #if self.verbose:
                #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

criterions = {
    "mse" : nn.MSELoss,
    "directional": DirectionalMeanSquaredError,
    "combined": CombinedLoss,
}
optimizers = {
    "adam": torch.optim.Adam,
}

class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = LSTMModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                              layer_dim=cfg["layer_dim"], output_dim=cfg["output_dim"],dropout=cfg["dropout"])
        self.cfg = cfg
        self.criterion = criterions[cfg["loss"]]()
        self.optimizer = optimizers[cfg["optim"]](self.model.parameters(), lr=cfg["training"]["learning_rate"],weight_decay=0.0005)
        self.epochs = cfg["training"]["num_epochs"]

        self.scaler = GradScaler()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=50)
        self.batch_size = 256
        self.model_saver = BestModelSaver(verbose=False,patience=999)

    def train(self, data_train: Tuple[np.ndarray, np.ndarray], data_valid: Tuple[np.ndarray, np.ndarray]) -> Tuple[list, list]:
        train_loss = []
        valid_loss = []
        inputs = torch.from_numpy(data_train[0]).float().to(__DEVICE__)
        targets = torch.from_numpy(data_train[1]).float().to(__DEVICE__)
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for epoch in pbar:#range(self.epochs):
            self.model.train()
            inputs, targets = inputs.to(__DEVICE__), targets.to(__DEVICE__)
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())
            valid_loss_epoch = self.validate((data_valid[0], data_valid[1]))
            valid_loss.append(valid_loss_epoch)
            if(epoch > 100):
                self.scheduler.step(valid_loss_epoch)
                # Best model saver
                self.model_saver(valid_loss_epoch, self.model, "checkpoint.pth")

            pbar.set_postfix({"Train Loss": loss.item (), "Valid Loss": valid_loss_epoch})
        # Load the last checkpoint with the best model
        self.model.load_state_dict(torch.load("checkpoint.pth"))
        self.save_model(prefix=f"{self.cfg['data']['window_size']}_Best_")
        return train_loss, valid_loss

    
    def predict(self, data:tuple):
        '''
        Predicts next price.
        '''
        with torch.no_grad():
            inputs = torch.from_numpy(data).float().to(__DEVICE__)
            outputs = self.model(inputs)
        return outputs

    def validate(self, valid_data: tuple) -> float:
        """
        Validates the model using the validation dataset.

        Parameters:
        valid_data (tuple): A tuple containing the validation inputs and targets.

        Returns:
        float: The validation loss.
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            inputs = torch.from_numpy(valid_data[0]).float().to(__DEVICE__)
            targets = torch.from_numpy(valid_data[1]).float().to(__DEVICE__)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        self.model.train()  # Set the model back to training mode
        return loss.item()


    def save_model(self,prefix=""):
        torch.save(self.model.state_dict(), f"models/{prefix}{self.cfg['name']}.pt")
        logging.info(f"Model saved to models/{prefix}{self.cfg['name']}.pt")
    def load_model(self, prefix=""):
        self.model.load_state_dict(torch.load(f"models/{prefix}{self.cfg['name']}.pt"))
        logging.info(f"Model loaded from models/{prefix}{self.cfg['name']}.pt")

