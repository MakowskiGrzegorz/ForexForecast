import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from src.fundamental_data import find_common_start_end_date
class DataLoader:
    def __init__(self, technical_path=None, fundamental_path=None):
        # Ładujemy dane z plików CSV
        if technical_path is not None:
            self.technical_data = pd.read_csv(technical_path)
            self.technical_data.dropna(inplace=True)


        if fundamental_path is not None:
            self.fundamental_data = pd.read_csv(fundamental_path)

        if technical_path and fundamental_path:
            start_dates = [self.technical_data['date'].min(), self.fundamental_data['date'].min()]
            end_dates = [self.technical_data['date'].max(), self.fundamental_data['date'].max()]
            start = max(start_dates)
            end = min(end_dates)
            
            self.technical_data = self.technical_data[self.technical_data['date'] >= start]
            self.technical_data = self.technical_data[self.technical_data['date'] <= end]

            self.fundamental_data = self.fundamental_data[self.fundamental_data['date'] >= start]
            self.fundamental_data = self.fundamental_data[self.fundamental_data['date'] <= end]

        self.technical_scaler = MinMaxScaler(feature_range=(0, 1))
        self.fundamental_scaler = MinMaxScaler(feature_range=(0, 1))

    
    def prepare_technical_data(self, scaling=True, window_size=20, overlap=True):
        technical_train = self.technical_data.iloc[:int(self.technical_data.shape[0]*0.8)]
        technical_valid = self.technical_data.iloc[int(self.technical_data.shape[0]*0.8):]
        technical_train.drop('date', axis=1, inplace=True)
        technical_valid.drop('date', axis=1, inplace=True)
        if scaling:
            technical_train = self.technical_scaler.fit_transform(technical_train)
            technical_valid = self.technical_scaler.fit_transform(technical_valid)
        X_train, Y_train = self.sliding_window_transform(technical_train,window_size=window_size, overlap=overlap)
        X_valid, Y_valid = self.sliding_window_transform(technical_valid,window_size=window_size, overlap=overlap)
        return (X_train, Y_train), (X_valid, Y_valid)

    def prepare_fundamental_data(self, scaling=True, window_size=20, overlap=True):
        fundamental_train = self.fundamental_data.iloc[:int(self.fundamental_data.shape[0]*0.8)]
        fundamental_valid = self.fundamental_data.iloc[int(self.fundamental_data.shape[0]*0.8):]
        fundamental_train.drop('date', axis=1, inplace=True)
        fundamental_valid.drop('date', axis=1, inplace=True)
        if scaling:
            fundamental_train = self.fundamental_scaler.fit_transform(fundamental_train)
            fundamental_valid = self.fundamental_scaler.fit_transform(fundamental_valid)
        X_train, Y_train = self.sliding_window_transform(fundamental_train,window_size=window_size, overlap=overlap)
        X_valid, Y_valid = self.sliding_window_transform(fundamental_valid,window_size=window_size, overlap=overlap)
        return (X_train, Y_train), (X_valid, Y_valid)


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
    
    def sliding_window_transform(self, data, window_size=5, overlap=True):
        X, Y = [], []
        if overlap:
            for i in range(len(data)-window_size): # overlapping windows 
                X.append(data[i:i+window_size])
                Y.append(data[i+window_size, 0]) # close is in first column
        else:
            for i in range(0, len(data)-window_size, window_size): # non-overlapping windows
                X.append(data[i:i+window_size])
                Y.append(data[i+window_size, 0])  # close is in first column
        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)
        print(X.shape, Y.shape)
        return X,Y

    
        

if __name__ == "__main__":
    dl = DataLoader('data/EURUSD240_technical.csv')
    print(dl.get_data())




