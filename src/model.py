import torch
from torch import nn
from src.lstm import LSTMModel, LSTMModelMultiStep, LSTMModelSeq2Seq
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.threshold import decision
import logging

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

__DEVICE__ = 'cuda' if torch.cuda.is_available() else 'cpu'
# class DirectionalMeanSquaredError(nn.Module): # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3560007
#     """
#     Klasa implementująca niestandardową funkcję straty, która minimalizuje różnicę
#     między przewidywaną a rzeczywistą wartością oraz dopasowuje różnicę do różnicy celu.
#     """
#     def __init__(self):
#         super(DirectionalMeanSquaredError, self).__init__()

#     def forward(self, predictions, targets):
#         """
#         Oblicza niestandardową stratę MSE uwzględniającą kierunek zmiany ceny.

#         :param predictions: Przewidywane wartości przez model.
#         :param targets: Rzeczywiste wartości.
#         :param target_diff: Rzeczywista różnica między kolejnymi wartościami w danych.
#         :return: Wartość straty.
#         """

#         pred_1 = torch.roll(predictions, 1)
#         target_1 = torch.roll(targets, 1)

#         sign = torch.sign((targets - target_1) * (predictions  - pred_1))

#         return torch.mean(torch.pow((predictions - targets) - 0.01*sign, 2))
class DirectionalMeanSquaredError(nn.Module):
    def __init__(self):
        super(DirectionalMeanSquaredError, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculates the custom MSE loss over a sequence of predictions and targets, considering the direction.

        :param predictions: Predicted values by the model, shape (batch_size, future_steps).
        :param targets: Actual target values, shape (batch_size, future_steps).
        :return: Loss value.
        """
        # Roll predictions and targets along the time axis
        if (targets.shape[1] == 1):
            pred_1 = torch.roll(predictions, 1)
            target_1 = torch.roll(targets, 1)
        else:
            pred_1 = torch.roll(predictions, 1, dims=1)
            target_1 = torch.roll(targets, 1, dims=1)

        # Compute directional sign
        sign = torch.sign((targets - target_1) * (predictions - pred_1))

        # Calculate the directional MSE
        loss = torch.mean(torch.pow((predictions - targets) - 0.01 * sign, 2))

        return loss

class SimpleDirectionalMeanSquaredError(nn.Module):
    def __init__(self):
        super(SimpleDirectionalMeanSquaredError, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculates the simple directional MSE loss over a sequence of predictions and targets.

        :param predictions: Predicted values by the model, shape (batch_size, future_steps).
        :param targets: Actual target values, shape (batch_size, future_steps).
        :return: Loss value.
        """
        # Compute differences between consecutive predictions and targets along the time axis
        if(targets.shape[1] == 1):
            pred_diff = predictions[1:] - predictions[:-1]
            target_diff = targets[1:] - targets[:-1]
        else:
            pred_diff = predictions[:, 1:] - predictions[:, :-1]
            target_diff = targets[:, 1:] - targets[:, :-1]

        # Calculate the squared error between these differences
        loss = torch.mean((pred_diff - target_diff) ** 2)

        return loss * 1000  # Scale factor to match original implementation


# class CombinedLoss(nn.Module):
#     def __init__(self, weight_mse=0.6, weight_directional=0.4):
#         super(CombinedLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.directional_loss = SimpleDirectionalMeanSquaredError()
#         self.weight_mse = weight_mse
#         self.weight_directional = weight_directional

#     def forward(self, predictions, targets):
#         """
#         Oblicza połączoną stratę, która jest ważoną sumą MSE i niestandardowej straty kierunkowej.

#         :param predictions: Przewidywane wartości przez model.
#         :param targets: Rzeczywiste wartości.
#         :return: Wartość połączonej straty.
#         """
#         # Standardowa strata MSE
#         loss_mse = self.mse_loss(predictions, targets)
#         # Niestandardowa strata kierunkowa
#         loss_directional = self.directional_loss(predictions, targets)
#         # Połączona strata z ważoną sumą
#         combined_loss = (self.weight_mse * loss_mse) + (self.weight_directional * loss_directional)
#         return combined_loss
class CombinedLoss(nn.Module):
    def __init__(self, weight_mse=0.6, weight_directional=0.4):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.directional_loss = SimpleDirectionalMeanSquaredError()
        self.weight_mse = weight_mse
        self.weight_directional = weight_directional

    def forward(self, predictions, targets):
        """
        Calculates the combined loss over a sequence of predictions and targets.

        :param predictions: Predicted values by the model, shape (batch_size, future_steps).
        :param targets: Actual target values, shape (batch_size, future_steps).
        :return: Combined loss value.
        """
        # Standard MSE loss over the sequence
        loss_mse = self.mse_loss(predictions, targets)

        # Directional loss over the sequence
        loss_directional = self.directional_loss(predictions, targets)

        # Combine the two losses with specified weights
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
    def __init__(self, model_cfg, data_cfg) -> None:
        super().__init__()

        future_steps = data_cfg["future_steps"]
        self.model = LSTMModelMultiStep(input_dim=model_cfg["input_dim"], hidden_dim=model_cfg["hidden_dim"],
                                        layer_dim=model_cfg["layer_dim"], output_dim=model_cfg["output_dim"],
                                        dropout=model_cfg["dropout"], future_steps=future_steps)
        # Or use LSTMModelSeq2Seq instead
        # self.model = LSTMModelSeq2Seq(input_dim=model_cfg["input_dim"], hidden_dim=model_cfg["hidden_dim"],
        #                               layer_dim=model_cfg["layer_dim"], output_dim=model_cfg["output_dim"],
        #                               dropout=model_cfg["dropout"], future_steps=future_steps)
        self.cfg = model_cfg
        self.data_cfg = data_cfg
        self.criterion = criterions[model_cfg["loss"]]()
        self.optimizer = optimizers[model_cfg["optim"]](self.model.parameters(), lr=model_cfg["training"]["learning_rate"],weight_decay=0.0005)
        self.epochs = model_cfg["training"]["num_epochs"]

        self.scaler = GradScaler()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=100,factor=0.05)
        self.model_saver = BestModelSaver(verbose=False,patience=999)

    def train(self, data_train: Tuple[np.ndarray, np.ndarray], data_valid: Tuple[np.ndarray, np.ndarray]) -> Tuple[list, list]:
        train_loss = []
        valid_loss = []
        inputs = torch.from_numpy(data_train[0]).float().to(__DEVICE__)
        targets = torch.from_numpy(data_train[1]).float().to(__DEVICE__)
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        batch_size = 128
        for epoch in pbar:#range(self.epochs):#
            self.model.train()
            train_loss_epoch = 0
            for i in range(0, inputs.size(0), batch_size):  # Iterate over batches
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
                
                batch_inputs, batch_targets = batch_inputs.to(__DEVICE__), batch_targets.to(__DEVICE__)
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(batch_inputs)
                    if len(outputs.shape) == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    loss = self.criterion(outputs, batch_targets)
                    train_loss_epoch += loss.item()
                    loss.backward()
                    self.optimizer.step()

            train_loss.append(train_loss_epoch/(inputs.size(0)/batch_size))
            valid_loss_epoch = self.validate((data_valid[0], data_valid[1]))
            valid_loss.append(valid_loss_epoch)
            if(epoch > 100):
                #self.scheduler.step(valid_loss_epoch)
                # Best model saver
                self.model_saver(valid_loss_epoch, self.model, "checkpoint.pth")

            pbar.set_postfix({"Train Loss": loss.item (), "Valid Loss": valid_loss_epoch, "Learning Rate": self.scheduler.get_last_lr()[-1], "Best loss":self.model_saver.val_loss_min})
        # Load the last checkpoint with the best model
        #self.model.load_state_dict(torch.load("checkpoint.pth"))
        self.save_model(prefix=f"{self.data_cfg['window_size']}_Best_")
        return train_loss, valid_loss

    
    def predict(self, data:tuple):
        '''
        Predicts next price.
        '''
        with torch.no_grad():
            inputs = torch.from_numpy(data).float().to(__DEVICE__)
            outputs = self.model(inputs)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(1)
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
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(1)
            loss = self.criterion(outputs, targets)
        self.model.train()  # Set the model back to training mode
        return loss.item()


    def save_model(self,prefix=""):
        torch.save(self.model.state_dict(), f"models/{prefix}{self.cfg['name']}.pt")
        logging.info(f"Model saved to models/{prefix}{self.cfg['name']}.pt")
    def load_model(self, prefix=""):
        self.model.load_state_dict(torch.load(f"models/{prefix}{self.cfg['name']}.pt"))
        logging.info(f"Model loaded from models/{prefix}{self.cfg['name']}.pt")

