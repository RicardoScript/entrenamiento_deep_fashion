import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# ================== 🔥 ACELERACIÓN GPU (NUEVO) ==================
# Usa FP16 en GPUs modernas → 1.5x–2x más rápido
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
# ================================================================

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV   = "data/processed/val.csv"
TEST_CSV  = "data/processed/test.csv"

MODEL_OUT_DIR = "models/deepfashion_mobilenetv2_savedmodel"
LABELS_JSON = "data/processed/labels.json"


def get_attr_cols(df):
    return [c for c in df.columns if c.startswith("attr_")]


# ================== 📦 DATA PIPELINE OPTIMIZADO ==================
def build_dataset(csv_path: str, training: bool):
    df = pd.read_csv(csv_path)
    attr_cols = get_attr_cols(df)

    paths = df["image_path"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    attrs = df[attr_cols].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, attrs))

    def _load(path, label, attrs):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, {"cat_out": label, "attr_out": attrs}

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    # 🔥 CACHE EN VAL/TEST (NUEVO) → evita releer disco cada epoch
    if not training:
        ds = ds.cache()

    # 🔥 DATA PIPELINE NO DETERMINISTA (NUEVO) → más rápido
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=min(len(paths), 5000),
                        seed=SEED,
                        reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
# ================================================================


def infer_num_classes():
    df = pd.read_csv(TRAIN_CSV)
    return int(df["label"].nunique())


def infer_num_attrs():
    df = pd.read_csv(TRAIN_CSV)
    return len(get_attr_cols(df))


# ================== ⚖️ LOSS PARA ATRIBUTOS DESBALANCEADOS ==================
# Penaliza más cuando el modelo falla en atributos positivos
def weighted_bce(y_true, y_pred):
    pos_weight = 10.0  # 🔥 NUEVO: fuerza aprender atributos raros
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * pos_weight + (1 - y_true)
    return tf.reduce_mean(bce * weights)
# ===========================================================================


def build_model(num_classes: int, num_attrs: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # 🔥 dtype=float32 evita problemas con mixed precision en salida softmax
    cat_out = tf.keras.layers.Dense(num_classes,
                                    activation="softmax",
                                    dtype="float32",
                                    name="cat_out")(x)

    attr_out = tf.keras.layers.Dense(num_attrs,
                                     activation="sigmoid",
                                     dtype="float32",
                                     name="attr_out")(x)

    model = tf.keras.Model(inputs, [cat_out, attr_out])
    return model, base


def main():
    tf.random.set_seed(SEED)

    num_classes = infer_num_classes()
    num_attrs = infer_num_attrs()

    train_ds = build_dataset(TRAIN_CSV, training=True)
    val_ds   = build_dataset(VAL_CSV, training=False)
    test_ds  = build_dataset(TEST_CSV, training=False)

    model, base = build_model(num_classes, num_attrs)

    # ================== 🧠 COMPILACIÓN FASE 1 ==================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "cat_out": tf.keras.losses.SparseCategoricalCrossentropy(),
            "attr_out": weighted_bce
        },
        # 🔥 NUEVO: los atributos pesan más en la pérdida total
        loss_weights={
            "cat_out": 1.0,
            "attr_out": 5.0
        },
        metrics={
            "cat_out": "accuracy",
            "attr_out": "binary_accuracy"
        }
    )

    # ================== 💾 CALLBACKS ROBUSTOS ==================
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),

        # 🔥 NUEVO: guarda el mejor modelo automáticamente
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("\n=== FASE 1 ===")
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    # ================== 🔓 FINE TUNING ESTABLE ==================
    base.trainable = True

    # 🔥 NUEVO: mantener BatchNorm congelado para estabilidad
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss={
            "cat_out": tf.keras.losses.SparseCategoricalCrossentropy(),
            "attr_out": weighted_bce
        },
        loss_weights={"cat_out": 1.0, "attr_out": 5.0},
        metrics={"cat_out": "accuracy", "attr_out": "binary_accuracy"}
    )

    print("\n=== FASE 2 ===")
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # ================== 🧪 EVALUACIÓN CORRECTA ==================
    print("\n=== TEST ===")
    results = model.evaluate(test_ds, return_dict=True)

    print("Test total loss:", results["loss"])
    print("Test cat_out_accuracy:", results["cat_out_accuracy"])
    print("Test attr_out_binary_accuracy:", results["attr_out_binary_accuracy"])

    print("\n=== Guardando SavedModel ===")
    os.makedirs(os.path.dirname(MODEL_OUT_DIR), exist_ok=True)
    model.save(MODEL_OUT_DIR)

    print("✅ Entrenamiento finalizado correctamente")


if __name__ == "__main__":
    main()
