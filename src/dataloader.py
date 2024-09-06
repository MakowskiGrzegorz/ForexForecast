import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from src.fundamental_data import find_common_start_end_date
class DataLoader:
    def __init__(self, technical_path=None, fundamental_path=None, threshold=None):
        # Ładujemy dane z plików CSV
        if technical_path is not None:
            self.technical_data = pd.read_csv(technical_path)
            self.technical_data.dropna(inplace=True)


        if fundamental_path is not None:
            self.fundamental_data = pd.read_csv(fundamental_path)
        if threshold is not None:
            self.threshold = pd.read_csv(threshold)

        if technical_path and fundamental_path:
            self.find_common_start_end()

        self.technical_scaler = StandardScaler()
        self.fundamental_scaler = StandardScaler()
        self.combined_scaler = StandardScaler()
        self.arimax_scaler = StandardScaler()
    def find_common_start_end(self):
        start_dates = [self.technical_data['date'].min(), self.fundamental_data['date'].min()]
        end_dates = [self.technical_data['date'].max(), self.fundamental_data['date'].max()]
        start = max(start_dates)
        end = min(end_dates)

        self.technical_data = self.technical_data[self.technical_data['date'] >= start]
        self.technical_data = self.technical_data[self.technical_data['date'] <= end]

        self.fundamental_data = self.fundamental_data[self.fundamental_data['date'] >= start]
        self.fundamental_data = self.fundamental_data[self.fundamental_data['date'] <= end]
        if hasattr(self, 'threshold'):
            self.threshold = self.threshold[self.threshold['date'].isin(self.fundamental_data['date'])]
            #self.threshold = self.threshold[self.threshold['date'] == self.fundamental_data['date']]
    def get_threshold_val(self):
        if hasattr(self, 'threshold'):
            data = self.threshold.drop('date',axis=1).to_numpy()
            threshold_train = data[:int(data.shape[0]*0.8)]
            threshold_valid = data[int(data.shape[0]*0.8):]
            return threshold_train, threshold_valid
        return None, None
    def prepare_technical_data(self, scaling=True, window_size=20,future_steps=1, overlap=True):
        data = self.technical_data.drop('date', axis=1).to_numpy()
        if scaling:
            data = self.technical_scaler.fit_transform(data)
        technical_train = data[:int(data.shape[0]*0.8)]
        technical_valid = data[int(data.shape[0]*0.8):]
        X_train, Y_train = self.sliding_window_transform(technical_train,window_size=window_size,future_steps=future_steps, overlap=overlap)
        X_valid, Y_valid = self.sliding_window_transform(technical_valid,window_size=window_size,future_steps=future_steps, overlap=overlap)
        return (X_train, Y_train), (X_valid, Y_valid)

    def prepare_fundamental_data(self, scaling=True, window_size=20,future_steps=1, overlap=True):
        data = self.fundamental_data.drop('date', axis=1).to_numpy()
        if scaling:
            data = self.fundamental_scaler.fit_transform(data)
        fundamental_train = data[:int(data.shape[0]*0.8)]
        fundamental_valid = data[int(data.shape[0]*0.8):]
        X_train, Y_train = self.sliding_window_transform(fundamental_train,window_size=window_size,future_steps=future_steps, overlap=overlap)
        X_valid, Y_valid = self.sliding_window_transform(fundamental_valid,window_size=window_size,future_steps=future_steps, overlap=overlap)
        return (X_train, Y_train), (X_valid, Y_valid)

    def prepare_combined_data(self, scaling=True, window_size=20,future_steps=1, overlap=True):
        data_technical = self.technical_data.drop('date',axis=1).to_numpy()
        data_fundamental = self.fundamental_data.drop(['date','close'], axis=1).to_numpy()
        data = np.concatenate([data_technical, data_fundamental], axis=1)
        if scaling:
            data = self.combined_scaler.fit_transform(data)
        combined_train = data[:int(data.shape[0]*0.8)]
        combined_valid = data[int(data.shape[0]*0.8):]
        X_train, Y_train = self.sliding_window_transform(combined_train,window_size=window_size,future_steps=future_steps, overlap=overlap)
        X_valid, Y_valid = self.sliding_window_transform(combined_valid,window_size=window_size,future_steps=future_steps, overlap=overlap)
        return (X_train, Y_train), (X_valid, Y_valid)
    def prepare_arimax_data(self, scaling=True):# TODO: please simplify this
        data_technical = self.technical_data.drop('date',axis=1).to_numpy()
        data_fundamental = self.fundamental_data.drop(['date','close'], axis=1).to_numpy()
        data = np.concatenate([data_technical, data_fundamental], axis=1)
        if scaling:
            data = self.arimax_scaler.fit_transform(data)

        combined_train = data[:int(data.shape[0]*0.8)]
        combined_valid = data[int(data.shape[0]*0.8):]
        return combined_train, combined_valid
    def inverse_scaling(self, data, type:str):
        features = data.shape[1]
        if type == "technical":
            if self.technical_data.shape[1] == features:
                return self.technical_scaler.inverse_transform(data)
            else:
                pad_data = np.zeros((data.shape[0], self.technical_data.shape[1]-1)) # we need to drop the date column
                pad_data[:,0:features] = data[:, 0:features]
                return self.technical_scaler.inverse_transform(pad_data)[:,0:features]
        elif type == "fundamental":
            if self.fundamental_data.shape[1] == features:
                return self.fundamental_scaler.inverse_transform(data)
            else:
                pad_data = np.zeros((data.shape[0], self.fundamental_data.shape[1]-1))# we need to drop the date column
                pad_data[:,0:features] = data[:, 0:features]
                return self.fundamental_scaler.inverse_transform(pad_data)[:,0:features]
    
    def sliding_window_transform(self, data, window_size=5, future_steps=1, overlap=True, step_size=1):
        """
        Transforms the dataset into a format suitable for time series forecasting with LSTM.
        
        Parameters:
        - data: The dataset, expected to be a 2D numpy array where rows are time steps and columns are features.
        - window_size: The size of the input window (number of past time steps to consider).
        - future_steps: The number of future time steps to predict.
        - overlap: Whether to use overlapping windows (default: True).
        - step_size: Step size for non-overlapping windows (default: 1).
        
        Returns:
        - X: Array of input sequences.
        - Y: Array of target sequences, predicting the next `future_steps` close prices.
        """
        X, Y = [], []
        
        if overlap:
            # Overlapping windows: Slide one step at a time
            for i in range(len(data) - window_size - future_steps + 1):
                X.append(data[i:i + window_size])
                Y.append(data[i + window_size:i + window_size + future_steps, 0])  # close price (first column) for next `future_steps`
        else:
            # Non-overlapping windows: Step by `step_size` or `window_size`
            for i in range(0, len(data) - window_size - future_steps + 1, step_size):
                X.append(data[i:i + window_size])
                Y.append(data[i + window_size:i + window_size + future_steps, 0])
        
        X = np.array(X)
        Y = np.array(Y).reshape(-1, future_steps)
        
        return X, Y


if __name__ == "__main__":
    dl = DataLoader('data/EURUSD240_technical.csv')
    dl.prepare_technical_data()




