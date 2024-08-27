import pandas as pd
import numpy as np

def generate_prices(output_file, start_date, end_date):
    # Wczytaj plik CSV


    # Wygeneruj nowe wiersze
    new_rows = pd.date_range(start=start_date, end=end_date, freq='H')
    new_df = pd.DataFrame(index=new_rows)

    # Wygeneruj losowe wartości dla kolumn 'open', 'high', 'low' i 'close'
    new_df['open'] = np.random.uniform(0, 3, size=len(new_df))
    new_df['high'] = new_df['open'] + np.random.uniform(0, 1, size=len(new_df))
    new_df['low'] = new_df['open'] - np.random.uniform(0, 1, size=len(new_df))
    new_df['close'] = new_df['open'] + np.random.uniform(-1, 1, size=len(new_df))


    # Zapisz wynik do nowego pliku CSV
    new_df.to_csv(output_file, sep=';')

def generate_indexes(output_file, start_date, end_date):
    # Wygeneruj nowe wiersze
    new_rows = pd.date_range(start=start_date, end=end_date, freq='M')
    new_df = pd.DataFrame(index=new_rows)

    # Wygeneruj losowe wartości dla kolumn 'inflation' i 'gdp'
    new_df['inflation'] = np.random.uniform(0, 10, size=len(new_df))
    new_df['gdp'] = np.random.uniform(1000, 10000, size=len(new_df))

    # Zapisz wynik do nowego pliku CSV
    new_df.to_csv(output_file, sep=';')


if __name__ == "__main__":
    generate_prices('data/prices.csv', '2021-01-01 00:00:00', '2021-05-30 23:00:00')
    generate_indexes('data/indexes.csv', '2021-01-01 00:00:00', '2021-05-31 23:00:00')

    