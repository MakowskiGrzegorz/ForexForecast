import pandas as pd


#TODO please double check if that calculation are correct or use a library for TA
def load_candlestick_data(filepath: str) -> pd.DataFrame:
    """

    Parametry:
        filepath (str): Ścieżka do pliku CSV z danymi.

    Zwraca:
        pd.DataFrame: DataFrame zawierający załadowane dane.
    """
    df = pd.read_csv(filepath, names=['date', 'open', 'high', 'low', 'close', 'volume'], parse_dates=['date'], index_col='date')
    return df

def calculate_moving_average(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Oblicza średnią przeglądową z danych.
    """
    data[f'MA{window_size}'] = round(data['close'].rolling(window=window_size).mean(), 5)
    return data

def calculate_macd(data: pd.DataFrame, window_short: int, window_long: int) -> pd.DataFrame:#, window_signal: int
    """
    Oblicza MACD z danych.
    """
    exp1 = data['close'].ewm(span=window_short, adjust=False).mean()
    exp2 = data['close'].ewm(span=window_long, adjust=False).mean()
    macd = exp1 - exp2
    data['MACD'] = round(macd,5)
    #data['MACD'] = calculate_moving_average(data, window_short) - calculate_moving_average(data, window_long)
    return data

def calculate_roc(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Oblicza ROC z danych.
    """
    data['ROC'] = round((data['close'] - data['close'].shift(window_size)) / data['close'].shift(window_size) * 100, 5)
    return data

def calculate_momentum(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Oblicza momentum z danych.
    """
    data['Momentum'] = round(data['close'].diff(window_size), 5)
    return data

def calculate_RSI(data: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Oblicza RSI z danych.
    """

        # get the price diff
    delta = data['close'].diff()

    # positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(alpha=1.0 / period, adjust=True).mean()
    _loss = down.abs().ewm(alpha=1.0 / period, adjust=True).mean()
    RS = _gain / _loss

    data["RSI"] = round(100 - (100 / (1 + RS)), 5)

    return data

def calculate_bolinger_bands(data: pd.DataFrame, window_size: int, deviations: int) -> pd.DataFrame:
    """
    Oblicza Bollinger Bands z danych.
    """

    data['Bollinger_Up'] = round(data['MA10'] + deviations * data['close'].rolling(window=window_size).std(), 5)
    data['Bollinger_Down'] = round(data['MA10'] - deviations * data['close'].rolling(window=window_size).std(), 5)
    return data

def calculate_cci(data: pd.DataFrame, window_size: int, deviations: int) -> pd.DataFrame:
    """
    Oblicza CCI z danych.
    """
    data['CCI'] = round((data['close'] - data['MA10']) / (deviations * data['close'].rolling(window=window_size).std()) * 100, 5)
    return data

def main():
    data = load_candlestick_data("data/EURUSD240.csv")
    data = data[['close']].copy()
    data = calculate_moving_average(data, 10)
    data = calculate_macd(data, 12, 26)
    data = calculate_roc(data, 2)
    data = calculate_momentum(data, 4)
    data = calculate_RSI(data, 10)
    data = calculate_bolinger_bands(data, 20, 2)
    data = calculate_cci(data, 20, 100)
    data.to_csv("data/EURUSD240_technical.csv")

    data['close'] = data['close'].diff().round(8).dropna()
    data.to_csv("data/EURUSD240_technical_diff.csv")


if __name__ == "__main__":
    main()

