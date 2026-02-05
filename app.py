import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_DIR = "models/deepfashion_mobilenetv2_savedmodel"
LABELS_JSON = "data/processed/labels.json"

st.set_page_config(
    page_title="Reconocimiento de Prendas",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    # Cargar SavedModel como capa de inferencia
    return TFSMLayer(
        MODEL_DIR,
        call_endpoint="serving_default"
    )

@st.cache_resource
def load_labels():
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

model = load_model()
labels = load_labels()

# ---------------- PREPROCESS ----------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

# ---------------- UI ----------------
st.title("Reconocimiento de Prendas")
st.write("Sube una imagen y el modelo entrenado la clasificara.")

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", width="stretch")

    if st.button("Clasificar"):
        with st.spinner("Analizando imagen..."):
            x = preprocess_image(image)

            # Inferencia
            output = model(x)

            # El SavedModel devuelve dict
            if isinstance(output, dict):
                output = list(output.values())[0]

            preds = output.numpy()[0]

            idx = int(np.argmax(preds))
            confidence = float(preds[idx])

            category = labels[idx] if idx < len(labels) else f"Clase {idx}"

        st.success("Clasificacion completada")
        st.markdown(f"### Categoria: **{category}**")
        st.markdown(f"### Confianza: **{confidence:.2%}**")
