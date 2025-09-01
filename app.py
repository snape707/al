import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from utils import load_labels, preprocess_pil_image

st.set_page_config(page_title="Alzheimer MRI Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Alzheimer MRI Detection (Local)")
st.write("Upload an MRI image to get a prediction. This runs **locally** on your machine.")

# Paths
MODEL_PATH = os.path.join("model", "alz_model.keras")
LABELS_PATH = os.path.join("model", "labels.json")
DEFAULT_IMG_SIZE = 224

@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        st.error("Model or labels not found. Train the model first (see README).")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)
    return model, labels

model, labels = load_model_and_labels()

img_file = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg", "jpeg", "png"])
img_size = st.number_input("Image size", value=DEFAULT_IMG_SIZE, min_value=128, max_value=384, step=32)

if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    arr = preprocess_pil_image(img, target_size=img_size)

    with st.spinner("Predicting..."):
        preds = model.predict(arr, verbose=0)[0]  # shape (num_classes,)
        idx = int(np.argmax(preds))
        cls = labels[idx]
        conf = float(preds[idx])

    st.subheader(f"Prediction: **{cls}**")
    st.write(f"Confidence: **{conf:.2%}**")

    # Show table of probabilities
    st.write("Class probabilities:")
    prob_rows = [{"class": labels[i], "probability": float(p)} for i, p in enumerate(preds)]
    st.dataframe(prob_rows, use_container_width=True)

st.markdown("---")
st.caption("Tip: For best results, train on your own dataset and keep images consistent with training pre-processing.")
