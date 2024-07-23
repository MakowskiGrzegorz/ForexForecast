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

