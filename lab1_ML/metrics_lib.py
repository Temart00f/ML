import numpy as np

# Метрики регрессии

# Коэффициент детерминации
def custom_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Средняя абсолютная ошибка
def custom_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Среднеквадратичная ошибка
def custom_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Корень из среднеквадратичной ошибки
def custom_rmse(y_true, y_pred):
    return np.sqrt(custom_mse(y_true, y_pred))

# Средняя абсолютная процентная ошибка
def custom_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))





# Метрики классификации

def custom_confusion_matrix(y_true, y_pred):
    """Возвращает матрицу ошибок [[TN, FP], [FN, TP]] для бинарной классификации."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return np.array([[TN, FP], [FN, TP]])

def custom_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def custom_precision(y_true, y_pred):
    cm = custom_confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def custom_recall(y_true, y_pred):
    cm = custom_confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def custom_f1(y_true, y_pred):
    p = custom_precision(y_true, y_pred)
    r = custom_recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0