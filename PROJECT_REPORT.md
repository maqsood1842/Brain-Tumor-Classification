# Brain Tumor Classification — Project Report

## Model and labels

The saved artifact is a MobileNetV2 Keras model with `224 × 224 × 3` RGB input and four softmax outputs. The original Colab notebook confirms the class order: `glioma`, `meningioma`, `notumor`, `pituitary` (indices 0–3). Preprocessing inside the model applies the MobileNetV2 `x / 127.5 - 1` normalization. Inference supplies raw 0–255 pixels.

The adjacent `step1/train.py` is a separate EfficientNetB0 experiment and is not the source of this model. Model weights were not modified or retrained for this work.

## Recorded Colab test results

These are the metrics recorded in the executed notebook, not a new local evaluation:

| Class | Precision | Recall | F1 score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9345 | 0.6775 | 0.7855 | 400 |
| Meningioma | 0.7763 | 0.5900 | 0.6705 | 400 |
| No Tumor | 0.8358 | 0.9800 | 0.9022 | 400 |
| Pituitary | 0.7412 | 0.9950 | 0.8495 | 400 |
| **Macro average** | **0.8219** | **0.8106** | **0.8019** | **1,600** |

- Test accuracy: **81.06%**
- Test loss: **0.6986**

`metrics.json` is transcribed from the notebook’s test report. `training_curves.png` and `confusion_matrix.png` are extracted from its embedded executed outputs. The confusion matrix sums to 1,600 examples and has 1,297 correct classifications, consistent with the reported accuracy.

The notebook’s final Drive directory listing includes `training_curves.png` and `confusion_matrix.png`; it does not include `metrics.json`. Google Drive is not mounted in this environment, so current Drive contents could not be independently checked. No metrics file is claimed to have been retrieved from Drive.

## Local verification and limitations

- Python syntax checks pass for the Streamlit app and inference utility.
- Metrics JSON structure and class ordering were checked.
- The embedded plots were opened and visually inspected.
- The local dataset copy has 5,600 training images (1,400 per class) and 1,600 test images, while the executed Colab notebook reports 5,601 training images. The local training copy therefore differs by one image from the notebook’s logged count.
- A local Streamlit run and per-image model inference were **not completed** because TensorFlow/Streamlit are not installed and the large TensorFlow wheel download could not be completed. The notebook records one sanity-check prediction: true `glioma`, predicted `notumor`, confidence **48.23%**. That is a notebook result, not a local app test.
- A screenshot of the running Streamlit interface remains outstanding; the saved notebook plots are available as report visuals.

## Responsible-use note

This is an educational classification demonstration, not a validated diagnostic system. Predictions and confidence scores must not be used for clinical decisions.
