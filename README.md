# CreditLens loan risk demo

A small Streamlit website that runs a published pretrained loan-default classifier. It does not train a model and does not require a GPU. The first prediction downloads `model.pkl` from Hugging Face and caches it locally.

https://creditlens-credit-risk.streamlit.app/

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. An internet connection is needed for the first model download.

## Model

- (https://huggingface.co/Yashwanth-R19/loan-default-classifier)
- Task: estimate loan default probability from tabular applicant and loan features
- Artifact: scikit-learn pipeline containing preprocessing and an XGBoost classifier
- Compatibility: scikit-learn is pinned to 1.5.2, the version used to serialize this pipeline
- Inference: CPU; this project does not train or fine-tune the model

The model card reports test ROC-AUC 0.746, precision 0.221, and recall 0.671 for its XGBoost model. These numbers are a reminder that an attractive demo is not a validated lending system. Do not use this output to approve or deny real applications. Validate on data matching the intended population, review calibration and fairness, and keep a human decision-maker involved.

The app collects data only in the current browser session and does not write applicant inputs to a database or file.
