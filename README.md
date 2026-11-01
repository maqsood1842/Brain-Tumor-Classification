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

The saved artifact is a Keras functional model with a MobileNetV2 (`mobilenetv2_1.00_224`) backbone and a four-unit softmax head. Its input shape is `224 × 224 × 3`. The dashboard converts uploaded images to RGB and resizes them to 224 × 224 with bilinear interpolation, matching the available training script's `image_dataset_from_directory` default. It passes 0–255 float pixels to the model; the saved graph contains the `x / 127.5 - 1` normalization.

The output labels used by the dashboard are:

1. Glioma (`glioma`)
2. Meningioma (`meningioma`)
3. No Tumor (`notumor`)
4. Pituitary (`pituitary`)

**Class-order provenance:** the available `step1/train.py` assigns `class_names = train_ds.class_names`; with the four current directory names, Keras's directory loader resolves the order to `glioma`, `meningioma`, `notumor`, `pituitary`. This matches the output order configured in `utils/inference.py`. However, that training script builds EfficientNetB0, while the saved artifact contains MobileNetV2. The original notebook and `class_names.json` are absent, so the script's verified label order cannot be conclusively attributed to these exact model weights. Obtain the matching training run's class names before deployment or interpreting predictions.

The model archive confirms a 4-neuron softmax output and embeds preprocessing equivalent to `x / 127.5 - 1`. The available training script and data loader both use 224 × 224 images and pass 0–255 pixel values to the model, consistent with the saved model's embedded normalization.

## Disclaimer

Predictions and confidence values are model outputs, not clinical findings or calibrated probabilities of disease. Image quality, acquisition differences, and training-data limitations can affect results. Do not use this application for clinical decisions.
