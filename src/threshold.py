import pandas as pd
import numpy as np
import math
from enum import Enum
def get_upper_threshold(close: pd.Series) -> float:
    """
    Oblicza górną granicę progu dla różnic cen zamknięcia.

    Argumenty:
    close (pd.Series): Seria danych zawierająca ceny zamknięcia.

    Zwraca:
    float: Górna granica progu.
    """
    # Oblicz różnicę między kolejnymi cenami zamknięcia, usuń brakujące wartości i weź wartości bezwzględne
    difference = close.diff().dropna().abs()

    # Podziel różnice na 10 przedziałów i policz wystąpienia w każdym przedziale
    bins = pd.cut(difference, bins=10)
    bins = bins.value_counts().to_frame().reset_index()
    # Zmień wartości w kolumnie 'close' na górną granicę przedziału
    bins["close"] = bins["close"].apply(lambda x: x.right)

    # Konwertuj DataFrame do tablicy numpy dla łatwiejszego dostępu do danych
    bins = bins.to_numpy()

    # Oblicz 85% liczby wszystkich różnic
    percentile_count = len(difference) * 0.85

    # Zainicjuj licznik sumujący liczby wystąpień w przedziałach
    count = 0
    # Iteruj przez wszystkie przedziały
    for i in range(10):
        count += bins[i, 1]
        # Jeśli suma przekroczy 85% wszystkich różnic, zwróć górną granicę bieżącego przedziału
        if count > percentile_count:
            return bins[i, 0]


def get_entropy(labels: list, base: float = None) -> float:
    """
    Oblicz entropię dla listy etykiet.

    Argumenty:
    labels (list): Lista etykiet, dla których ma być obliczona entropia.
    base (float, opcjonalnie): Podstawa logarytmiczna do użycia. Domyślnie logarytm naturalny (e).

    Zwraca:
    float: Wartość entropii.
    """
    value_counts = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = math.e if base is None else base
    entropy = -(value_counts * np.log(value_counts) / np.log(base)).sum()
    return entropy

def get_threshold(close: pd.Series) -> float:
    """
    Oblicza najlepszy próg dla różnic cen zamknięcia.

    Argumenty:
    close (pd.Series): Seria danych zawierająca ceny zamknięcia.

    Zwraca:
    float: Najlepszy próg.
    """
    # Oblicz różnice między kolejnymi cenami zamknięcia i usuń brakujące wartości
    difference = close.diff().dropna().tolist()

    # Inicjalizacja początkowej wartości progu
    threshold = 0.0
    # Pobierz górną granicę progu z funkcji pomocniczej
    thres_upper_bound = get_upper_threshold(close)
    # Inicjalizacja tymczasowego progu do iteracji
    temp_thres = 0.0
    # Inicjalizacja zmiennej do przechowywania najlepszej wartości entropii
    best_entropy = -float('inf')

    # Iteruj przez możliwe wartości progu aż do osiągnięcia górnej granicy
    while temp_thres < thres_upper_bound:
        # Przypisz etykiety na podstawie porównania z tymczasowym progiem
        labels = [2 if diff > temp_thres else 1 if -diff > temp_thres else 0 for diff in difference]
        # Oblicz entropię dla przypisanych etykiet
        entropy = get_entropy(labels)
        # Aktualizuj najlepszy próg jeśli znaleziono lepszą entropię
        if entropy > best_entropy:
            best_entropy = entropy
            threshold = temp_thres
        # Inkrementuj tymczasowy próg
        temp_thres += 0.00001

    # Zwróć najlepszy znaleziony próg
    return threshold



class Decision(Enum):
    INC = 1
    DEC = 2
    NOACT = 3


def decision(diff: float, threshold: float) -> Decision:
    """
    Decyduje o akcji na podstawie różnicy i progu.

    Argumenty:
    diff (float): Różnica, która ma być oceniona.
    threshold (float): Próg, który decyduje o wyniku.

    Zwraca:
    Decision: INC jeśli różnica jest większa niż próg, DEC jeśli negatywna różnica jest większa niż próg, NOACT w przeciwnym razie.
    """
    if diff > threshold:
        return Decision.INC
    elif -diff > threshold:
        return Decision.DEC
    else:
        return Decision.NOACT
def calculate_3_month_thresholds(loader):
    # Extract 'date' and 'close' columns
    close_with_date = loader.technical_data[['date', 'close']].copy()

    # Ensure that the 'date' column is a datetime type
    close_with_date['date'] = pd.to_datetime(close_with_date['date'])

    # Set the 'date' column as the index
    close_with_date.set_index('date', inplace=True)

    # Resample data into 3-month periods and forward-fill the values
    df_3_monthly = close_with_date.resample('3M').apply(lambda x: get_threshold(x['close']))

    # Dictionary to hold the 3-month thresholds
    three_month_thresholds = {}
    previous_threshold = 0
    
    for period, threshold in df_3_monthly.items():
        # Store the previous threshold value
        three_month_thresholds[period] = previous_threshold
        print(f"3-Month threshold value for {period}: {previous_threshold}")
        previous_threshold = threshold

    return three_month_thresholds

def calculate_monthly_thresholds(loader):
    # Extract 'date' and 'close' columns
    close_with_date = loader.technical_data[['date', 'close']].copy()

    # Ensure that the 'date' column is a datetime type
    close_with_date['date'] = pd.to_datetime(close_with_date['date'])

    # Group the data by year and month
    grouped = close_with_date.groupby(close_with_date['date'].dt.to_period('M'))

    # Dictionary to hold the monthly thresholds
    monthly_thresholds = {}
    previous_threeshold = 0
    for period, df in grouped:
        # Calculate the threshold value for the entire month
        monthly_threshold_value = get_threshold(df['close'])
        monthly_thresholds[period] = previous_threeshold
        print(f"Monthly threshold value for {period}: {previous_threeshold}")
        previous_threeshold = monthly_threshold_value

    return monthly_thresholds
def save_thresholds_to_csv(monthly_thresholds, output_file):
    # Create a time range with 4-hour intervals covering all months
    start_date = min([period.start_time for period in monthly_thresholds.keys()])
    end_date = max([period.end_time for period in monthly_thresholds.keys()]) + pd.DateOffset(months=1)
    time_range = pd.date_range(start=start_date, end=end_date, freq='4H')

    # Create a DataFrame to hold the interpolated threshold values
    interpolated_df = pd.DataFrame({'4h_period': time_range})

    # Map each 4-hour period to the corresponding monthly threshold value
    def map_threshold(period):
        month = period.to_period('M')
        return monthly_thresholds.get(month, None)

    interpolated_df['monthly_threshold_value'] = interpolated_df['4h_period'].apply(lambda x: map_threshold(x))

    # Save the DataFrame to a CSV file
    interpolated_df.to_csv(output_file, index=False)
    print(f"4-hour interval thresholds saved to {output_file}")
def load_and_resample_csv(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the 'date' column as the index
    df.set_index('date', inplace=True)

    # Sort the DataFrame by index (date)
    df.sort_index(inplace=True)

    # Resample the data to 4-hour intervals, using forward-fill to fill missing values
    df_resampled = df.resample('4H').ffill()

    # Save the resampled DataFrame to a new CSV file
    df_resampled.to_csv(output_file)
    print(f"4-hour interval data saved to {output_file}")
if __name__ == "__main__":
    from src.dataloader import DataLoader
    load_and_resample_csv('data/3_months.csv', 'data/resampled_output.csv')
    # loader = DataLoader('data/EURUSD240_technical.csv', "data/fundamental_data.csv")
    # monthly_thresholds = calculate_3_month_thresholds(loader)
    # save_thresholds_to_csv(monthly_thresholds, output_file="data/1_month_thresholds.csv")
