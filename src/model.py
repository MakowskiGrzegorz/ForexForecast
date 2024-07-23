import torch
from torch import nn
from src.lstm import LSTMModel
from tqdm import tqdm
from typing import Tuple
import numpy as np
from src.threshold import decision
import logging
class DirectionalMeanSquaredError(nn.Module):
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
        # Oblicz różnicę między przewidywaniami a rzeczywistymi wartościami
        pred_diff = predictions[1:] - predictions[:-1]
        target_diff = targets[1:] - targets[:-1]
        # Oblicz błąd kwadratowy między przewidywaną a rzeczywistą różnicą
        loss = torch.mean((pred_diff - target_diff) ** 2)
        return loss * 1000


class CombinedLoss(nn.Module):
    def __init__(self, weight_mse=0.35, weight_directional=0.65):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.directional_loss = DirectionalMeanSquaredError()
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


criterions = {
    "mse" : nn.MSELoss,
    "directional": DirectionalMeanSquaredError,
    "combined": CombinedLoss
}
optimizer = {
    "sgd" : torch.optim.SGD,
    "adam": torch.optim.Adam,
}

class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.lstm = LSTMModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                              layer_dim=cfg["layer_dim"], output_dim=cfg["output_dim"])
        self.cfg = cfg
        self.criterion = criterions[cfg["loss"]]()
        self.optimizer = optimizer[cfg["optim"]](self.lstm.parameters(), lr=cfg["training"]["learning_rate"])
        self.epochs = cfg["training"]["num_epochs"]
        
    def train(self, data_train: Tuple[np.ndarray, np.ndarray], data_valid: Tuple[np.ndarray, np.ndarray]) -> Tuple[list, list]:
        """
        Trenuje model LSTM używając podanych danych treningowych i walidacyjnych.

        Parametry:
            data_train (Tuple[np.ndarray, np.ndarray]): Krotka zawierająca dane treningowe (wejścia i cele).
            data_valid (Tuple[np.ndarray, np.ndarray]): Krotka zawierająca dane walidacyjne (wejścia i cele).

        Zwraca:
            Tuple[list, list]: Krotka zawierająca listy strat treningowych i walidacyjnych dla każdej epoki.
        """
        inputs = torch.from_numpy(data_train[0]).float()
        targets = torch.from_numpy(data_train[1]).float()
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        train_loss = []
        valid_loss = []
        for epoch in pbar:
            outputs = self.lstm(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())

            valid_loss_epoch = self.validate((data_valid[0], data_valid[1]))
            valid_loss.append(valid_loss_epoch)
            
            pbar.set_postfix({"Train Loss": loss.item(), "Valid Loss": valid_loss_epoch})
            if epoch > 0 and epoch % 10 == 0:
                self.save_model(prefix=f"Epoch_{epoch}")
        return train_loss, valid_loss
        
    def predict(self, data:tuple):
        '''
        Predicts next price.
        '''
        with torch.no_grad():
            inputs = torch.from_numpy(data).float()
            outputs = self.lstm(inputs)
        return outputs

    def validate(self, valid_data: tuple) -> float:
        """
        Validates the model using the validation dataset.

        Parameters:
        valid_data (tuple): A tuple containing the validation inputs and targets.

        Returns:
        float: The validation loss.
        """
        self.lstm.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            inputs = torch.from_numpy(valid_data[0]).float()
            targets = torch.from_numpy(valid_data[1]).float()
            outputs = self.lstm(inputs)
            loss = self.criterion(outputs, targets)
        self.lstm.train()  # Set the model back to training mode
        return loss.item()


    def save_model(self,prefix=""):
        torch.save(self.lstm.state_dict(), f"models/{prefix}{self.cfg['name']}.pt")
        logging.info(f"Model saved to models/{prefix}{self.cfg['name']}.pt")
    def load_model(self, prefix=""):
        self.lstm.load_state_dict(torch.load(f"models/{prefix}{self.cfg['name']}.pt"))
        logging.info(f"Model loaded from models/{prefix}{self.cfg['name']}.pt")

