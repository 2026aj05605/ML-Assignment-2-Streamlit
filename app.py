"""A saved-model evaluation dashboard. Launch: streamlit run app.py."""
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay
from core import evaluate, validate_data, METRICS

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CellScope | Model comparison', page_icon='🔬', layout='wide')

@st.cache_resource
def load_artifacts():
    manifest = json.loads((ROOT / 'model' / 'manifest.json').read_text(encoding='utf-8'))
    models = {name: joblib.load(ROOT / 'model' / file) for name, file in manifest['model_files'].items()}
    return manifest, models

st.title('🔬 CellScope')
st.caption('Five classifiers · one holdout dataset · transparent evaluation')
st.info('Educational experiment using UCI cell-nucleus measurements. Not a clinical decision tool.')
try:
    manifest, models = load_artifacts()
except Exception as exc:
    st.error(f'Unable to load saved models: {exc}. Run python train.py in the project folder.')
    st.stop()

with st.sidebar:
    st.header('Evaluation controls')
    selected = st.selectbox('Choose a classifier', list(models))
    uploaded = st.file_uploader('Upload labelled test data (CSV)', type=['csv'])
    st.download_button('Download sample test CSV', (ROOT / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
    st.caption('Required: all 30 features + target. target: 0 = benign; 1 = malignant. Only saved models are used.')

try:
    if uploaded is not None and uploaded.size > 5 * 1024 * 1024:
        raise ValueError('Please upload a test CSV smaller than 5 MB.')
    frame = pd.read_csv(uploaded if uploaded is not None else ROOT / 'test_data.csv')
    if len(frame) > 10000:
        raise ValueError('Please upload at most 10,000 test rows.')
    X, y = validate_data(frame, manifest['features'])
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader('Evaluation dataset')
st.write(f"{'Uploaded CSV' if uploaded is not None else 'Bundled held-out test set'} · {len(X)} rows · {X.shape[1]} features · {int(y.sum())} malignant")
st.caption('Bundled split: 455 training / 114 test rows, stratified, seed 42. Uploaded rows are evaluated as supplied; use unseen data for valid generalization estimates.')
if y.nunique() < 2:
    st.warning('This CSV contains one class. AUC is undefined and is displayed as N/A. Precision/recall/F1 use zero when their denominator is zero; MCC follows scikit-learn conventions.')
all_scores, predictions = [], {}
for name, model in models.items():
    scores, predicted, probability = evaluate(model, X, y)
    all_scores.append({'Model': name, **scores})
    predictions[name] = (predicted, probability)
comparison = pd.DataFrame(all_scores).set_index('Model')
st.subheader(f'{selected} · current test data')
for col, metric in zip(st.columns(6), METRICS):
    value = comparison.loc[selected, metric]
    col.metric(metric, 'N/A' if pd.isna(value) else f'{value:.4f}')
st.subheader('All five models')
st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'), width='stretch')
st.download_button('Download comparison', comparison.to_csv().encode(), 'model_comparison.csv', 'text/csv')

predicted, probability = predictions[selected]
left, right = st.columns(2)
with left:
    st.subheader('Confusion matrix')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, predicted, labels=[0, 1], display_labels=['Benign', 'Malignant'], cmap='Blues', colorbar=False, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader('ROC curve')
    if y.nunique() == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y, probability, ax=ax, name=selected)
        ax.plot([0, 1], [0, 1], '--', color='gray')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write('ROC requires examples of both classes.')
with st.expander('Classification report', expanded=True):
    report = classification_report(y, predicted, labels=[0, 1], target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
with st.expander('Inspect data and predictions'):
    result = frame.copy()
    result['prediction'] = predicted
    result['malignant_probability'] = probability
    st.dataframe(result, width='stretch')
    st.download_button('Download predictions', result.to_csv(index=False).encode(), 'predictions.csv', 'text/csv')
st.caption('AUC uses malignant-class probabilities. Precision, recall and F1 are binary scores for malignant=1; MCC uses both classes. Scaling is part of each saved pipeline where applicable.')
