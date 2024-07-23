import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from threshold import decision

def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Oblicza błąd średniokwadratowy (RMSE) między prognozami a rzeczywistymi wartościami.

    Parametry:
        predictions (np.ndarray): Tablica numpy zawierająca prognozowane wartości.
        targets (np.ndarray): Tablica numpy zawierająca rzeczywiste wartości.

    Zwraca:
        float: Obliczony RMSE.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_smape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Oblicza symetryczny średni procentowy błąd bezwzględny (sMAPE) między prognozami a rzeczywistymi wartościami.

    Parametry:
        predictions (np.ndarray): Tablica numpy zawierająca prognozowane wartości.
        targets (np.ndarray): Tablica numpy zawierająca rzeczywiste wartości.

    Zwraca:
        float: Obliczony sMAPE.
    """

    return 100/len(targets) * np.sum(2 * np.abs(predictions - targets) / (np.abs(targets) + np.abs(predictions)))





def count_classification_results(predictions: np.ndarray, targets: np.ndarray) -> (int, int, int):
    """
    Liczy prawdziwie pozytywne, fałszywie pozytywne i fałszywie negatywne wyniki.

    Parametry:
        predictions (np.ndarray): Tablica numpy zawierająca prognozowane klasyfikacje.
        targets (np.ndarray): Tablica numpy zawierająca rzeczywiste klasyfikacje.

    Zwraca:
        tuple: Krotka zawierająca liczbę prawdziwie pozytywnych, fałszywie pozytywnych i fałszywie negatywnych wyników.
    """
    true_positives = np.sum((predictions == 1) & (targets == 1))
    false_positives = np.sum((predictions == 1) & (targets == 2))
    false_negatives = np.sum((predictions == 2) & (targets == 1))

    return true_positives, false_positives, false_negatives


def calculate_f1_score(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """
    Oblicza miarę F1-score.

    Parametry:
        true_positives (int): Liczba prawdziwie pozytywnych wyników.
        false_positives (int): Liczba fałszywie pozytywnych wyników.
        false_negatives (int): Liczba fałszywie negatywnych wyników.

    Zwraca:
        float: Obliczony F1-score.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def calculate_auc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Oblicza wartość AUC (Area Under the Curve) dla krzywej ROC.

    Parametry:
        predictions (np.ndarray): Tablica numpy zawierająca prognozowane prawdopodobieństwa.
        targets (np.ndarray): Tablica numpy zawierająca rzeczywiste klasyfikacje (0 lub 1).

    Zwraca:
        float: Obliczona wartość AUC.
    """



    # Binarize the output for multilabel classification
    classes = np.unique(targets)
    targets_binarized = label_binarize(targets, classes=classes)
    predictions_binarized = label_binarize(predictions, classes=classes)
    
    return roc_auc_score(targets_binarized, predictions_binarized, average='macro', multi_class='ovr')





def evaluate_model(model_name:str, predictions:np.ndarray, targets:np.ndarray, threeshold_val:float):
    print(f"Model: {model_name}")
    print(f"RMSE: {calculate_rmse(predictions, targets)}")
    print(f"sMAPE: {calculate_smape(predictions, targets)}")


    predict_diff = np.diff(predictions, axis=0)
    target_diff = np.diff(targets, axis=0)
    predict_decision = np.apply_along_axis(lambda x: int(decision(x, threeshold_val).value), 1, predict_diff)
    target_decision = np.apply_along_axis(lambda x: int(decision(x, threeshold_val).value), 1, target_diff)
    print(f"AUC: {calculate_auc(target_decision, predict_decision)}")
    print(f"F1-score: {calculate_f1_score(*count_classification_results(predict_decision, target_decision))}")
