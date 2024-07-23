import pandas as pd
from typing import List, Dict, Tuple
def preprocess_index_data(idx_daily_path: str, idx_240_path: str=None) -> pd.DataFrame:
    """
    Przetwarza dane indeksów giełdowych z plików CSV, łącząc dane dzienne i 240-minutowe (jeśli dostępne),
    a następnie wykonuje szereg operacji czyszczących i normalizujących.

    Parametry:
        idx_daily_path (str): Ścieżka do pliku CSV zawierającego dane dzienne indeksu.
        idx_240_path (str, opcjonalnie): Ścieżka do pliku CSV zawierającego dane indeksu z okresu 240 minut.
                                         Jeśli nie podano, używane są tylko dane dzienne.

    Zwraca:
        pd.DataFrame: DataFrame zawierający przetworzone dane indeksu, z kolumnami po czyszczeniu i normalizacji,
                      gdzie indeksem jest czas, a dane są próbkowane co 4 godziny.

    Przetwarzanie obejmuje:
    - Konwersję kolumny 'Time' na typ datetime.
    - Usunięcie niepotrzebnych kolumn ('Change', '%Chg', 'Volume', 'Symbol', 'Open Int').
    - Zmianę nazwy kolumny 'Last' na 'Close'.
    - Wypełnienie brakujących wartości metodą 'ffill' (forward fill).
    - Ustawienie kolumny 'Time' jako indeksu.
    - Próbkowanie danych co 4 godziny z wypełnieniem metodą 'ffill'.
    """
    idx_daily = pd.read_csv(idx_daily_path)
    idx_daily['Time'] = pd.to_datetime(idx_daily['Time'])

    if idx_240_path is not None:
        idx_240 = pd.read_csv(idx_240_path)
        idx_240['Time'] = pd.to_datetime(idx_240['Time'])
    else:
        idx_240 = pd.DataFrame()

    combined_data = pd.concat([idx_240, idx_daily])
    combined_data.drop("Change", axis=1, inplace=True)
    combined_data.drop("%Chg", axis=1, inplace=True)
    combined_data.drop("Volume", axis=1, inplace=True)
    combined_data.drop("Symbol", axis=1, inplace=True, errors="ignore")
    combined_data.drop("Open Int", axis=1, inplace=True, errors="ignore")
    combined_data.rename(columns={'Last': 'Close'}, inplace=True)
    
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.set_index('Time', inplace=True)
    combined_data = combined_data.resample('4H').ffill()
    return combined_data[['Close']]

def preprocess_single_rate(interest_rate_path: str) -> pd.DataFrame:
    """
    Przetwarza dane dotyczące stóp procentowych z pliku CSV, konwertując daty na typ datetime,
    ustawiając czas jako indeks i resamplując dane do 4-godzinnych interwałów.

    Parametry:
        interest_rate_path (str): Ścieżka do pliku CSV zawierającego dane o stopach procentowych.

    Zwraca:
        pd.DataFrame: DataFrame zawierający przetworzone dane o stopach procentowych,
                      z czasem jako indeksem i próbkowanym co 4 godziny.
                      
    Przetwarzanie obejmuje:
    - Zmianę nazwy kolumny 'DATE' na 'Time'.
    - Konwersję kolumny 'Time' na typ datetime.
    - Ustawienie kolumny 'Time' jako indeksu DataFrame.
    - Konwersję wartości stóp procentowych na liczby (z obsługą błędów).
    - Wypełnienie brakujących wartości metodą 'ffill' (forward fill).
    - Próbkowanie danych co 4 godziny z wypełnieniem metodą 'ffill'.
    """
    interest_rate = pd.read_csv(interest_rate_path, delimiter=';')
    interest_rate.rename(columns={'DATE': 'Time'}, inplace=True)
    interest_rate['Time'] = pd.to_datetime(interest_rate['Time'])
    interest_rate.set_index('Time', inplace=True)

    column_names = interest_rate.columns.tolist()
    
    interest_rate[column_names[0]] = pd.to_numeric(interest_rate[column_names[0]], errors='coerce')

    interest_rate.fillna(method='ffill', inplace=True)

    interest_rate = interest_rate.resample('4H').ffill()

    return interest_rate

def find_common_start_end_date(dataframes: List[pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Znajduje wspólną datę początkową i końcową dla listy ram danych.

    Parametry:
        dataframes (list[pd.DataFrame]): Lista ram danych, dla których ma zostać znaleziona wspólna data początkowa i końcowa.

    Zwraca:
        tuple: Krotka zawierająca wspólną datę początkową i końcową (common_start_date, common_end_date).

    Każda ramka danych powinna mieć indeks typu datetime.
    """
    #start_dates = [df['dat']]
    start_dates = [df.index.min() for df in dataframes]
    end_dates = [df.index.max() for df in dataframes]

    common_start_date = max(start_dates)
    common_end_date = min(end_dates)

    return common_start_date, common_end_date

def prepare_fundamental_data(dataframes: Dict[str, pd.DataFrame], start, end) -> None:
    fundamental = pd.DataFrame()
    for name, df in dataframes.items():
        df = df.loc[start:end]
        fundamental[name] = df.iloc[:,0]

    fundamental.to_csv("data/fundamental_data.csv")

def main():

    data = {}
    df = pd.read_csv("data/EURUSD240.csv", names=['date', 'open', 'high', 'low', 'close', 'volume'], parse_dates=['date'], index_col='date')
    data["close"] = df[["close"]]

    us_stock_index = preprocess_index_data("data/spx_1d.csv", "data/spx_240.csv")
    data['us_stock_index'] = us_stock_index
    eu_stock_index = preprocess_index_data("data/fxm24_1d.csv", "data/fxm24_240.csv")
    data['eu_stock_index'] = eu_stock_index
    de_stock_index = preprocess_index_data("data/dax_1d.csv")
    data['de_stock_index'] = de_stock_index

    eu_interest_rate = preprocess_single_rate("data/interest_rate_eu.csv")
    data['eu_interest_rate'] = eu_interest_rate
    de_interest_rate = preprocess_single_rate("data/interest_rate_ger.csv")
    data['de_interest_rate'] = de_interest_rate
    
    us_inflation_rate = preprocess_single_rate("data/inflation_rate_us.csv")
    data['us_inflation_rate'] = us_inflation_rate
    eu_inflation_rate = preprocess_single_rate("data/inflation_rate_eu.csv")
    data['eu_inflation_rate'] = eu_inflation_rate
    fed_exchange_rate = preprocess_single_rate("data/federal_exchange_rate.csv")
    data['fed_exchange_rate'] = fed_exchange_rate
    
    start,end = find_common_start_end_date(data.values())
    
    prepare_fundamental_data(data, start, end)

    

    #print(dax_data.head(20))
    #print(dax_data.tail(20))

if __name__ == "__main__":
    main()

