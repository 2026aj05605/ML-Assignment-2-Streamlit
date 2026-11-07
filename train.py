"""Reproduce a fixed, stratified holdout experiment and save complete pipelines."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from core import evaluate

ROOT = Path(__file__).resolve().parent
SEED = 42
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}

def main():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # sklearn uses malignant=0; explicitly invert so malignant is the positive class.
    y = (data.target == 0).astype(int).rename('target')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    (ROOT / 'model').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    pd.concat([X, y], axis=1).to_csv(ROOT / 'dataset.csv', index=False)
    pd.concat([X_train, y_train], axis=1).to_csv(ROOT / 'train_data.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
        'Decision Tree': make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', probability=True, random_state=SEED)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, cv_rows, details = [], [], {}
    for name, model in models.items():
        auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_rows.append({'Model': name, 'Mean train-CV AUC': auc.mean(), 'Std train-CV AUC': auc.std()})
        model.fit(X_train, y_train)
        scores, predicted, _ = evaluate(model, X_test, y_test)
        rows.append({'Model': name, **scores})
        details[name] = {
            'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
            'classification_report': classification_report(y_test, predicted, labels=[0, 1], target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0),
            'parameters': str(model),
        }
        joblib.dump(model, ROOT / 'model' / MODEL_FILES[name])
    pd.DataFrame(rows).to_csv(ROOT / 'results' / 'metrics.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(ROOT / 'results' / 'cross_validation.csv', index=False)
    (ROOT / 'results' / 'details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    manifest = {
        'features': list(X.columns), 'target': 'target', 'labels': {'0': 'Benign', '1': 'Malignant'},
        'model_files': MODEL_FILES, 'seed': SEED, 'train_size': len(X_train), 'test_size': len(X_test),
        'train_indices': X_train.index.tolist(), 'test_indices': X_test.index.tolist(),
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'training_cv_winner': max(cv_rows, key=lambda r: r['Mean train-CV AUC'])['Model'],
    }
    (ROOT / 'model' / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
