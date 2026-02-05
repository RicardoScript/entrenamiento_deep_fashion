import os
import json
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io

IMG_SIZE = 224
MODEL_DIR = "models/deepfashion_mobilenetv2_savedmodel"
LABELS_JSON = "data/processed/labels.json"

app = FastAPI(title="DeepFashion Classifier API", version="1.0")

model = None
labels = None


def load_labels():
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data
    return None


@app.on_event("startup")
def startup():
    global model, labels
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(f"No existe MODEL_DIR: {MODEL_DIR}. Entrena y guarda el modelo primero.")
    model = tf.keras.models.load_model(MODEL_DIR)
    labels = load_labels()
    print("✅ Modelo cargado:", MODEL_DIR)
    if labels:
        print("✅ Labels cargados:", len(labels))


def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype(np.float32)

    # Preprocess MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    return x


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, labels

    t0 = time.time()
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))  # noqa
    except Exception:
        return JSONResponse(status_code=400, content={"error": "No se pudo leer la imagen"})

    x = preprocess_pil(img)

    preds = model.predict(x, verbose=0)[0]  # shape (num_classes,)
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    category = labels[idx] if labels and idx < len(labels) else str(idx)

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "category_id": idx,
        "category": category,
        "confidence": conf,
        "processing_time_ms": elapsed_ms
    }