import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV   = "data/processed/val.csv"
TEST_CSV  = "data/processed/test.csv"

MODEL_OUT_DIR = "models/deepfashion_mobilenetv2_savedmodel"  # SavedModel folder
LABELS_JSON = "data/processed/labels.json"  # lista index->nombre (opcional, pero recomendado)


def build_dataset(csv_path: str, training: bool):
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} debe tener columnas: image_path,label")

    paths = df["image_path"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        # Leer imagen
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)  # si son png, funciona igual la mayoría de casos
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)

        # Preprocess MobileNetV2
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # Augmentations básicas
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def infer_num_classes():
    # Saca #clases desde train.csv
    df = pd.read_csv(TRAIN_CSV)
    return int(df["label"].nunique())


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # fase 1: congelado

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    tf.random.set_seed(SEED)

    num_classes = infer_num_classes()
    print("Num classes:", num_classes)

    train_ds = build_dataset(TRAIN_CSV, training=True)
    val_ds   = build_dataset(VAL_CSV, training=False)
    test_ds  = build_dataset(TEST_CSV, training=False)

    model, base = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]

    print("\n=== FASE 1: entrenar cabeza ===")
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    print("\n=== FASE 2: fine-tuning (últimas capas) ===")
    base.trainable = True
    # Ajusta cuántas capas descongelas:
    fine_tune_at = max(0, len(base.layers) - 50)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    print("\n=== Evaluación en test ===")
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test loss:", test_loss)
    print("Test acc :", test_acc)

    print("\n=== Guardando SavedModel ===")
    os.makedirs(os.path.dirname(MODEL_OUT_DIR), exist_ok=True)
    model.save(MODEL_OUT_DIR)  # SavedModel (carpeta)

    # (Opcional) Validar labels.json existe
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, list) or len(labels) != num_classes:
            print("⚠️ labels.json existe pero no coincide con num_classes.")
        else:
            print("labels.json OK:", len(labels), "labels")

    print("✅ Listo. Modelo en:", MODEL_OUT_DIR)


if __name__ == "__main__":
    main()