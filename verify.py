"""Integration checks: saved models, metrics, schema failures, and Streamlit UI."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from core import validate_data, evaluate

ROOT = Path(__file__).resolve().parent

def main():
    # Hand-checkable example: TN=FP=FN=TP=1, and 3/4 positive-negative
    # probability pairs are correctly ordered. This independently checks AUC.
    class ReferenceModel:
        classes_ = np.array([0, 1])
        def predict(self, X):
            return np.array([0, 1, 0, 1])
        def predict_proba(self, X):
            p = np.array([0.1, 0.8, 0.4, 0.9])
            return np.column_stack([1-p, p])
    known, _, _ = evaluate(ReferenceModel(), pd.DataFrame({'x': range(4)}), pd.Series([0, 0, 1, 1]))
    assert np.allclose(list(known.values()), [0.5, 0.75, 0.5, 0.5, 0.5, 0.0])
    manifest = json.loads((ROOT / 'model/manifest.json').read_text())
    data = pd.read_csv(ROOT / 'test_data.csv')
    X, y = validate_data(data, manifest['features'])
    assert len(X) == 114 and len(manifest['train_indices']) == 455
    assert not set(manifest['train_indices']) & set(manifest['test_indices'])
    recorded = pd.read_csv(ROOT / 'results/metrics.csv').set_index('Model')
    for name, file in manifest['model_files'].items():
        model = joblib.load(ROOT / 'model' / file)
        scores, predicted, probability = evaluate(model, X, y)
        assert np.allclose(list(scores.values()), recorded.loc[name].values, atol=1e-12)
        assert len(predicted) == len(X) and np.isfinite(probability).all()
        single = y == 0
        assert np.isnan(evaluate(model, X[single], y[single])[0]['AUC'])
    bad_inputs = [data.drop(columns=[manifest['features'][0]]), data.assign(target=2),
                  data.assign(extra=1), data.iloc[:0], data.assign(target=np.nan)]
    bad = data.copy()
    bad.iloc[0, 0] = np.inf
    bad_inputs.append(bad)
    for frame in bad_inputs:
        try:
            validate_data(frame, manifest['features'])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid CSV accepted')
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception, str(app.exception)
    assert len(app.metric) == 6
    for name in manifest['model_files']:
        app.selectbox[0].select(name).run()
        assert not app.exception, str(app.exception)
        expected = recorded.loc[name]
        for widget, metric in zip(app.metric, expected.index):
            assert widget.value == f'{expected[metric]:.4f}'
    summary = 'PASS: hand-calculated metric fixture; five saved models reproduce all six metrics; disjoint train/test rows; six invalid CSV cases rejected; single-class AUC handled; Streamlit renders and all five model selections show correct metrics.'
    (ROOT / 'results/verification.txt').write_text(summary + '\n', encoding='utf-8')
    print(summary)

if __name__ == '__main__':
    main()
