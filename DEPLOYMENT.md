# Finish the account-dependent steps

This package contains a locally runnable implementation. It is not yet a published
submission. Do not claim the local execution happened on BITS Virtual Lab.

## 1. Run on BITS Virtual Lab

Open your assigned BITS lab, copy/extract this project, and open a terminal there.
Use Python 3.12. From the project folder run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python verify.py
python -m streamlit run app.py
```

Open the displayed app URL. Upload test_data.csv, select each model, and inspect
the results. Capture ONE genuine screenshot showing execution in the BITS lab,
including visible lab context, the app, model selection and metrics. Save it as
`bits_lab_execution.png` in this folder. A local screenshot is not lab evidence.

## 2. Publish the repository

Create a GitHub repository in your own account. Upload this folder's contents at
the repository root, including the five model/*.joblib files, manifest, CSVs,
README, requirements, app.py, core.py, train.py and verify.py. Do not upload your
virtual environment. Make the repository accessible to the evaluator. Preserve
honest commit history as you review, understand and personalize the work.

## 3. Streamlit Community Cloud

Open https://share.streamlit.io/ and sign in with GitHub. Create an app, select the
repository and branch, set entrypoint to `app.py`, and choose Python 3.12 in the
advanced settings. Deploy. The runtime loads the included models; it does not
run train.py or require data downloads. Visit the public URL in a private browser
window, upload test_data.csv, and check all five models.

If changing scikit-learn, numpy or pandas versions, retrain and verify before
deploying; serialized model compatibility across versions is not guaranteed.

## 4. Create the submission PDF

After the links and real screenshot exist, run:

```bash
python -m pip install reportlab
python make_submission.py --repo-url https://github.com/YOUR_ACCOUNT/YOUR_REPO --app-url https://YOUR_APP.streamlit.app --screenshot bits_lab_execution.png
```

This updates README section c and generates `submission.pdf` in the required
order: GitHub link, live app link, BITS lab screenshot, and full README content.
The generator checks URL formats and screenshot readability, but does not
verify remote accessibility or whether the screenshot is genuinely from BITS.
Review the output and open both links before submitting. Do not submit the
local-results preview as a finished assignment.

The assignment permits AI for learning support only. Read and understand every
part, make your own justified changes, and follow your course's attribution rules.
