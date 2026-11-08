"""Small dependency-free binary classification metric helpers."""

import numpy as np


def binary_metrics(y_true, y_score, threshold=50.0):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fpr = fp / max(1, fp + tn)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "false_positive_rate": fpr, "n": int(len(y_true)),
            "threshold": threshold}

