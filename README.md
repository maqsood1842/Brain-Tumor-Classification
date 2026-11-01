# NeuroVision — Brain Tumor Classification Dashboard

A Streamlit dashboard for demonstrating the existing brain MRI classification model. It provides an image upload and preview, a predict action, confidence, and a probability chart for four model outputs.

> **Educational use only.** This project is not a medical device and must not be used to diagnose, screen for, rule out, or treat a medical condition.

## Features

- Modern blue-and-white interface with sidebar navigation
- Home dashboard, MRI classification, project overview, and model details
- JPG, JPEG, and PNG upload with preview
- Explicit prediction action and all-class probability chart
- Loads the existing Keras model; no model training or dataset download is required

## Project files

```text
app.py
brain_tumor_model.keras
requirements.txt
utils/
  __init__.py
  inference.py
```

Keep `brain_tumor_model.keras` beside `app.py`. The model is intentionally retained in the project; datasets and virtual environments should not be committed.

## Run locally

Python 3.10–3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local URL after startup (normally `http://localhost:8501`).

## Model and preprocessing

The saved artifact is a Keras functional model with a MobileNetV2 (`mobilenetv2_1.00_224`) backbone and a four-unit softmax head. Its input shape is `224 × 224 × 3`. The dashboard converts uploaded images to RGB and resizes them to 224 × 224 with Pillow's default bicubic interpolation, matching the existing app's resize behavior. It passes 0–255 float pixels to the model; the saved graph contains the `x / 127.5 - 1` normalization.

The output labels used by the dashboard are:

1. Glioma (`glioma`)
2. Meningioma (`meningioma`)
3. No Tumor (`notumor`)
4. Pituitary (`pituitary`)

**Class-order limitation:** the original `class_names.json` was not present with the model. The order above follows the alphabetical class-folder order in the available dataset and was approved as an inference for this dashboard. Since the nearby training script describes a different backbone (EfficientNetB0), this mapping is not independently verified against the exact run that produced the saved artifact. Check the original training-run labels before relying on model output.

## Disclaimer

Predictions and confidence values are model outputs, not clinical findings or calibrated probabilities of disease. Image quality, acquisition differences, and training-data limitations can affect results. Do not use this application for clinical decisions.
