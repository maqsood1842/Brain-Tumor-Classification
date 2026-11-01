from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}
IMAGE_SIZE = (224, 224)


def preprocess_image(image: Image.Image) -> tf.Tensor:
    """Resize to the training loader's RGB input format; the model handles scaling."""
    resized = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    return tf.convert_to_tensor(np.expand_dims(pixels, axis=0), dtype=tf.float32)


def predict_image(model: tf.keras.Model, image: Image.Image) -> np.ndarray:
    """Return one validated probability score for each configured output class."""
    predictions = np.asarray(model.predict(preprocess_image(image), verbose=0)[0], dtype=np.float64)
    if predictions.shape != (len(CLASS_NAMES),):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} prediction scores, received shape {predictions.shape}."
        )
    if not np.all(np.isfinite(predictions)) or np.any(predictions < 0):
        raise ValueError("The model returned invalid probability scores.")
    if not np.isclose(predictions.sum(), 1.0, atol=1e-3):
        raise ValueError("The model scores do not sum to 1; the output is not a valid probability distribution.")
    return predictions
