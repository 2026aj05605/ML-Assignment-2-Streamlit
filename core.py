"""Shared schema validation and evaluation; never fits a model."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

def validate_data(frame, features):
    required = features + ['target']
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(required))
    if missing or extra:
        raise ValueError(f'CSV schema mismatch. Missing: {missing}. Unexpected: {extra}.')
    if frame.empty:
        raise ValueError('CSV must contain at least one data row.')
    try:
        numeric = frame[required].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('All feature values and target must be numeric.') from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Missing or infinite values are not supported; provide complete finite data.')
    if not numeric.target.isin([0, 1]).all():
        raise ValueError('target must contain only 0 (benign) and 1 (malignant).')
    return numeric[features], numeric.target.astype(int)

def evaluate(model, X, y):
    predicted = model.predict(X)
    positive_index = list(model.classes_).index(1)
    probability = model.predict_proba(X)[:, positive_index]
    scores = dict(zip(METRICS, [
        accuracy_score(y, predicted),
        roc_auc_score(y, probability) if y.nunique() == 2 else np.nan,
        precision_score(y, predicted, zero_division=0),
        recall_score(y, predicted, zero_division=0),
        f1_score(y, predicted, zero_division=0),
        matthews_corrcoef(y, predicted),
    ]))
    return scores, predicted, probability
