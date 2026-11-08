# CellScope - Five-model classification laboratory

## a. Problem statement

Compare five supervised classifiers for distinguishing malignant from benign breast-mass samples using numeric cell-nucleus measurements. Build a Streamlit dashboard that evaluates saved models on labelled test CSVs without retraining. This is an educational benchmark, not a clinical decision system.

## b. Dataset description

Dataset: UCI Breast Cancer Wisconsin (Diagnostic), donated by Wolberg, Mangasarian and Street (1995).
Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
DOI: https://doi.org/10.24432/C5DW2B
The project uses the offline copy bundled with scikit-learn's load_breast_cancer. No external download is required to train or run.

There are 569 records, 30 numeric input features, 212 malignant cases and 357 benign cases. This exceeds the assignment's 500-record and 12-feature minimums. The 30 inputs represent mean, standard error and worst measurements of ten cell-nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Identifier columns are excluded. This copy has no missing values.

Target convention: target=0 means benign and target=1 means malignant. This deliberately reverses scikit-learn's original encoding, so malignant is the positive class.

Input feature names, in training order:
mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension.

Evaluation uses a single stratified 80:20 split, random_state=42: 455 training rows and 114 unseen test rows (72 benign, 42 malignant). Split row indices are saved in model/manifest.json. All five models use identical rows. The holdout is used for final reporting, not parameter tuning. Five-fold stratified cross-validation on training data provides a separate AUC comparison in results/cross_validation.csv. Cross-validation scaling is fitted inside each fold through pipelines. No test-set resampling or synthetic data are used.

## c. Github Repository Link

GitHub repository: https://github.com/2026aj05605/ML-Assignment-2-Streamlit
Live Streamlit application: https://2026aj05605-ml-assignment-2.streamlit.app

## d. Models used

Logistic Regression: StandardScaler followed by L2 LogisticRegression(C=1, max_iter=3000, random_state=42).
Decision Tree: DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42), without scaling.
KNN: StandardScaler followed by KNeighborsClassifier(n_neighbors=5, uniform weights, Euclidean distance).
Naive Bayes: StandardScaler followed by GaussianNB(var_smoothing=1e-9).
Support Vector Machine: StandardScaler followed by SVC(C=1, kernel=rbf, gamma=scale, probability=True, random_state=42).

These are fixed baseline choices. No hyperparameter search is performed. Scaling is unnecessary for tree splits, but makes feature units comparable for the distance, margin and linear models. GaussianNB also uses the shared standardized representation. Each serialized pipeline includes all required preprocessing. SVC probability estimation uses internal training-data calibration; no test labels enter fitting.

### Comparison table - held-out test data

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Accuracy is the fraction correct. AUC is ROC AUC from malignant-class probabilities, not hard class predictions. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R). These three scores use malignant=1 as the positive class. MCC=(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). The app follows scikit-learn's zero-denominator convention: precision/recall/F1 and degenerate MCC return zero; AUC is explicitly N/A when only one class is present. Models use their fitted predict rules; no test-data threshold tuning is performed.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.9649; malignant recall 0.9286; false negatives 3, false positives 1. The standardized linear decision boundary is effective here; regularization controls coefficient size. |
| Decision Tree | Accuracy 0.9035; malignant recall 0.7857; false negatives 9, false positives 2. Depth 5 and minimum leaf size 3 constrain overfitting. Axis-aligned splits can miss smooth multivariate boundaries. |
| KNN | Accuracy 0.9561; malignant recall 0.9048; false negatives 4, false positives 1. Five-neighbor voting benefits from scaling. Correlated dimensions and local sample density can affect distances. |
| Naive Bayes | Accuracy 0.9211; malignant recall 0.8571; false negatives 6, false positives 3. Gaussian likelihoods provide a simple baseline. Strongly correlated nucleus measurements violate conditional independence. |
| Support Vector Machine | Accuracy 0.9737; malignant recall 0.9286; false negatives 3, false positives 0. The scaled RBF kernel can capture nonlinear separation. C=1 and gamma=scale are fixed defaults rather than test-tuned choices. |
| Overall winner for this dataset? | On this fixed holdout, Support Vector Machine leads by malignant F1 (AUC breaks ties): F1 0.9630, AUC 0.9947. Training-only five-fold CV selects Logistic Regression by mean AUC. Rankings are conditional on the split and metric. |

These explanations describe plausible model behavior rather than proving causation. With only 114 test cases, small differences may reflect sampling variation. Malignant recall matters because a false negative misses a malignant example, while precision reflects unnecessary positive flags. No conclusion about clinical readiness or population-wide performance follows from this benchmark. A larger external cohort and validation of calibration would be needed for that purpose.

## Run locally or on BITS Virtual Lab

Use Python 3.12. Open a terminal in this folder:

```bash
python -m venv .venv
# Windows activation:
.venv\Scripts\activate
# Linux/macOS activation instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python verify.py
python -m streamlit run app.py
```

Saved models are already included. Run train.py when reproducing the experiment or changing package versions; ordinary app use does not require training. Main package versions are pinned in requirements.txt. requirements-lock.txt records the full tested dependency set. Do not load joblib files obtained from untrusted sources. The app loads only the five packaged files, never uploaded models.

## Using the app

The bundled test data is evaluated initially. Download its CSV from the sidebar to see the exact schema. Upload only labelled test data with the same 30 numeric feature columns plus target; column order is normalized automatically. Missing/extra columns, non-numeric values, absent values, infinite values, invalid targets and empty datasets are rejected with clear messages. Uploads are limited to 5 MB and 10,000 rows. Single-class uploads are supported with AUC marked N/A. Reusing training rows gives optimistic metrics; the app cannot establish the provenance of arbitrary uploads.

The dropdown selects the detailed model view. All five results remain visible in the comparison table on the same uploaded rows. Six metric cards, confusion matrix, ROC curve and classification report explain the selected model. Download comparison scores and per-row predictions as CSV. The app never fits models or preprocessing on uploaded data.

## Project contents

app.py: interactive Streamlit frontend.
core.py: shared CSV validation and metric calculation.
train.py: reproducible training, split export, CV and saved artifacts.
verify.py: integration checks for loaded models, metrics, invalid inputs and all dropdown selections.
model/: five .joblib pipelines and a schema/split manifest.
dataset.csv, train_data.csv, test_data.csv: full source copy and the exact experiment split.
results/: holdout metrics, training CV results, confusion matrices, classification reports and verification record.
requirements.txt, requirements-lock.txt: package specifications.
DEPLOYMENT.md: BITS lab, GitHub, Streamlit and final PDF instructions.
make_submission.py: PDF assembly from this README, deployment URLs and authentic lab screenshot; requires reportlab separately.
