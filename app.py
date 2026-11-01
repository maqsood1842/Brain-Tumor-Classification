from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from utils.inference import CLASS_NAMES, DISPLAY_NAMES, predict_image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "brain_tumor_model.keras"
PAGES = ("Home", "Classify MRI", "About Project", "Model Details")
MODEL_INPUT_SIZE = (224, 224)
MODEL_FILE_SIZE_MB = MODEL_PATH.stat().st_size / (1024 * 1024) if MODEL_PATH.exists() else None

st.set_page_config(
    page_title="NeuroVision | Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --blue: #1769aa;
        --blue-dark: #12558a;
        --sky: #eaf5ff;
        --ink: #20364b;
        --muted: #64788b;
        --line: #dce8f1;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(216, 239, 255, .72), transparent 34rem),
            #f5f9fc;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(245, 249, 252, .82); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #153e62 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p { color: #e8f4fc; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: .3rem .5rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: .6rem .25rem 1.4rem;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: rgba(255,255,255,.14);
        font-size: 24px;
    }
    .brand-title { color: #fff; font-size: 1.1rem; font-weight: 750; line-height: 1.2; }
    .brand-subtitle { color: #b9d8ed; font-size: .73rem; margin-top: 3px; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.7rem, 4vw, 3rem);
        margin: .25rem 0 1.35rem;
        border: 1px solid #d6eafa;
        border-radius: 24px;
        background: linear-gradient(115deg, #fff 0%, #f3faff 63%, #e6f4ff 100%);
        box-shadow: 0 14px 36px rgba(27, 74, 112, .07);
    }
    .hero:after {
        content: "＋";
        position: absolute;
        right: 4%;
        top: -38%;
        color: rgba(23,105,170,.07);
        font-size: 18rem;
        font-weight: 200;
        line-height: 1;
        pointer-events: none;
    }
    .eyebrow {
        color: #1769aa;
        font-size: .74rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #102a43;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.1;
        letter-spacing: -.04em;
        margin: .65rem 0 .75rem;
    }
    .hero p { color: #526d82; font-size: 1.04rem; max-width: 700px; margin: 0; }
    .section-heading { color: #102a43; font-size: 1.25rem; font-weight: 720; margin: .4rem 0 .3rem; }
    .section-copy { color: #64788b; font-size: .94rem; margin: 0 0 .9rem; }
    .info-card {
        height: 100%;
        padding: 1.2rem 1.25rem;
        border: 1px solid #dfebf3;
        border-radius: 17px;
        background: rgba(255,255,255,.9);
        box-shadow: 0 8px 24px rgba(27,74,112,.04);
    }
    .info-card h3 { color: #173c59; font-size: 1rem; margin: .15rem 0 .45rem; }
    .info-card p { color: #64788b; font-size: .91rem; line-height: 1.55; margin: 0; }
    .info-icon { font-size: 1.35rem; margin-bottom: .45rem; }
    .stButton > button[kind="primary"] {
        border: 0;
        border-radius: 11px;
        background: linear-gradient(100deg, #1769aa, #2584c6);
        box-shadow: 0 7px 18px rgba(23,105,170,.18);
        font-weight: 650;
    }
    .stButton > button[kind="primary"]:hover { background: #12558a; }
    div[data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid #dfebf3;
        border-radius: 15px;
        background: #fff;
        box-shadow: 0 6px 18px rgba(27,74,112,.035);
    }
    div[data-testid="stMetricLabel"] { color: #64788b; }
    div[data-testid="stMetricValue"] { color: #173c59; }
    .fine-print { color: #718397; font-size: .82rem; }
    @media (max-width: 640px) {
        .hero { border-radius: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model() -> tf.keras.Model:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.input_shape[1:] != (*MODEL_INPUT_SIZE, 3):
        raise ValueError(f"Expected model input shape (*, 224, 224, 3), got {model.input_shape}.")
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs for the configured classes, "
            f"got {model.output_shape[-1]}."
        )
    return model


def show_hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_home() -> None:
    show_hero(
        "Brain MRI · Educational AI",
        "A clearer view of your model.",
        "Explore a trained image-classification model through a clean, guided MRI analysis experience.",
    )
    metrics = st.columns(4)
    metrics[0].metric("Classification groups", "04")
    metrics[1].metric("Model input", "224 × 224")
    metrics[2].metric("Backbone", "MobileNetV2")
    metrics[3].metric(
        "Model artifact",
        f"{MODEL_FILE_SIZE_MB:.1f} MB" if MODEL_FILE_SIZE_MB is not None else "Not found",
    )

    st.markdown('<div class="section-heading">Your workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A simple workflow from image selection to model output.</div>',
        unsafe_allow_html=True,
    )
    cards = st.columns(3)
    card_content = (
        ("◉", "Add an MRI image", "Upload a JPG, JPEG, or PNG image and review it before analysis."),
        ("⌁", "Run the classifier", "Request a prediction from the saved model—no retraining is performed."),
        ("▤", "Review all scores", "Compare the predicted class with the probability scores for all four outputs."),
    )
    for column, (icon, title, copy) in zip(cards, card_content):
        with column:
            st.markdown(
                f'<div class="info-card"><div class="info-icon">{icon}</div>'
                f"<h3>{title}</h3><p>{copy}</p></div>",
                unsafe_allow_html=True,
            )

    st.info(
        "For learning and demonstration only. This application is not a medical device and "
        "cannot diagnose, rule out, or guide treatment for any condition."
    )


def show_classifier() -> None:
    show_hero(
        "Image analysis",
        "Classify an MRI",
        "Choose a brain MRI image, inspect the preview, then run the saved model to view its four-class scores.",
    )
    upload_column, guide_column = st.columns([1.45, 1], gap="large")
    with upload_column:
        uploaded = st.file_uploader(
            "Choose an MRI image",
            type=("jpg", "jpeg", "png"),
            help="Supported formats: JPG, JPEG, PNG. Images are processed in this session only.",
        )

    with guide_column:
        st.markdown(
            '<div class="info-card"><div class="info-icon">ⓘ</div>'
            "<h3>Before you begin</h3>"
            "<p>Use an MRI image similar to the model’s training images. The output is a model "
            "estimate for an educational demonstration, not a medical conclusion.</p></div>",
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Upload an image to see its preview and enable prediction.")
        return

    raw_image = uploaded.getvalue()
    image_digest = hashlib.sha256(raw_image).hexdigest()
    try:
        with Image.open(io.BytesIO(raw_image)) as opened_image:
            image = opened_image.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"The selected file could not be decoded as an image: {error}")
        return

    preview_column, action_column = st.columns([1.05, 1], gap="large")
    with preview_column:
        st.image(image, caption=f"{uploaded.name} · {image.width} × {image.height}", use_container_width=True)
    with action_column:
        st.markdown('<div class="section-heading">Ready for analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Your image is converted to RGB and resized to 224 × 224 '
            "using Pillow’s default bicubic interpolation. Pixel scaling is applied inside the trained model.</div>",
            unsafe_allow_html=True,
        )
        if not MODEL_PATH.is_file():
            st.error(f"Model file not found: `{MODEL_PATH.name}`. Place it beside `app.py`.")
            return
        run_prediction = st.button(
            "Analyze MRI",
            type="primary",
            use_container_width=True,
        )
        if run_prediction:
            try:
                with st.spinner("Running the trained model..."):
                    model = load_model()
                    probabilities = predict_image(model, image)
            except (OSError, ValueError) as error:
                st.error(f"Unable to run the model: {error}")
                return
            st.session_state["prediction"] = {
                "digest": image_digest,
                "probabilities": probabilities.tolist(),
            }

    saved = st.session_state.get("prediction")
    if not saved or saved["digest"] != image_digest:
        return

    probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
    predicted_index = int(np.argmax(probabilities))
    predicted_name = CLASS_NAMES[predicted_index]
    predicted_label = DISPLAY_NAMES[predicted_name]
    confidence = float(probabilities[predicted_index])

    st.divider()
    st.markdown('<div class="section-heading">Model result</div>', unsafe_allow_html=True)
    result_columns = st.columns([1, 1.2], gap="large")
    with result_columns[0]:
        st.metric("Predicted class", predicted_label)
        st.metric("Model confidence", f"{confidence:.1%}")
        if confidence < 0.70:
            st.warning("The model score is relatively low. Review the image quality and treat this result cautiously.")
        st.caption("Confidence is the model’s output score, not a clinical certainty.")
    with result_columns[1]:
        probability_frame = pd.DataFrame(
            {
                "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
                "Probability": probabilities,
            }
        )
        st.markdown("#### Probability by class")
        st.bar_chart(
            probability_frame,
            x="Probability",
            y="Class",
            horizontal=True,
            height=250,
        )
        st.dataframe(
            probability_frame.assign(Probability=probability_frame["Probability"].map(lambda value: f"{value:.1%}")),
            hide_index=True,
            use_container_width=True,
        )


def show_about() -> None:
    show_hero(
        "Project overview",
        "About this project",
        "A university-project demonstration of image classification with a previously trained Keras model.",
    )
    first, second = st.columns(2, gap="large")
    with first:
        st.markdown(
            '<div class="info-card"><div class="info-icon">🎯</div><h3>Project purpose</h3>'
            "<p>This interface makes the existing model easier to demonstrate: upload an image, "
            "request a prediction, and inspect the scores across Glioma, Meningioma, No Tumor, "
            "and Pituitary.</p></div>",
            unsafe_allow_html=True,
        )
    with second:
        st.markdown(
            '<div class="info-card"><div class="info-icon">🛡️</div><h3>Responsible use</h3>'
            "<p>This is an educational prototype, not a validated diagnostic system. It may make "
            "incorrect predictions and should not be used for screening, diagnosis, treatment, "
            "or other clinical decisions.</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown("#### How to use the dashboard")
    st.markdown(
        "1. Open **Classify MRI** from the sidebar.\n"
        "2. Upload a supported image and check the preview.\n"
        "3. Select **Analyze MRI** to run the stored model.\n"
        "4. Review the predicted class, confidence score, and all four class probabilities."
    )
    st.caption(
        "The model artifact is used as provided. This project does not train, fine-tune, or replace it."
    )


def show_model_details() -> None:
    show_hero(
        "Technical notes",
        "Model information",
        "Architecture and inference details read from the saved Keras artifact and the available project files.",
    )
    model_column, input_column = st.columns(2, gap="large")
    with model_column:
        st.markdown("#### Architecture")
        st.markdown(
            "- **Framework:** TensorFlow / Keras\n"
            "- **Backbone:** MobileNetV2 (`mobilenetv2_1.00_224`)\n"
            "- **Classifier head:** global average pooling → dropout (0.3) → 4-unit softmax\n"
            "- **Input tensor:** `224 × 224 × 3` (`float32`)\n"
            "- **Artifact:** `brain_tumor_model.keras`"
        )
    with input_column:
        st.markdown("#### Inference preprocessing")
        st.markdown(
            "- Convert the uploaded image to RGB.\n"
            "- Resize to `224 × 224` with Pillow's default bicubic interpolation, matching the existing app.\n"
            "- Supply 0–255 pixel values; the model graph applies `x / 127.5 - 1`.\n"
            "- Model output: four softmax scores."
        )

    st.markdown("#### Output classes")
    st.dataframe(
        pd.DataFrame(
            {
                "Model output index": range(len(CLASS_NAMES)),
                "Dataset folder name": CLASS_NAMES,
                "Dashboard label": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "Class-order provenance: the original `class_names.json` is not present beside the model. "
        "This app uses the alphabetical order of the available training dataset folders "
        "(glioma, meningioma, notumor, pituitary), as approved for this implementation. "
        "Verify this mapping against the original training run before interpreting any output."
    )
    st.caption(
        "The adjacent training script describes an EfficientNetB0 model, whereas this saved artifact "
        "contains MobileNetV2. The script therefore cannot independently verify this artifact’s class mapping."
    )


st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-mark">🧠</div>
        <div><div class="brand-title">NeuroVision</div>
        <div class="brand-subtitle">MRI CLASSIFICATION STUDIO</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("#### Workspace")
selected_page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Educational prototype · Not for clinical use")

if selected_page == "Home":
    show_home()
elif selected_page == "Classify MRI":
    show_classifier()
elif selected_page == "About Project":
    show_about()
else:
    show_model_details()

st.markdown(
    '<div class="fine-print" style="text-align:center;padding:1.2rem 0 .5rem;">'
    "NeuroVision · Educational project only · Not a medical device</div>",
    unsafe_allow_html=True,
)
