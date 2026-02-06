# prepare_deepfashion_csv.py
import os
import pandas as pd

ROOT_IMG = "img"  # carpeta con imágenes
ANNO_DIR = "anno"

LIST_CATEGORY_IMG = os.path.join(ANNO_DIR, "list_category_img.txt")
LIST_EVAL_PARTITION = os.path.join(ANNO_DIR, "list_eval_partition.txt")
LIST_CATEGORY_CLOTH = os.path.join(ANNO_DIR, "list_category_cloth.txt")  # opcional
LIST_ATTR_IMG = os.path.join(ANNO_DIR, "list_attr_img.txt")
LIST_ATTR_CLOTH = os.path.join(ANNO_DIR, "list_attr_cloth.txt")

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# AGREGAR CARGA DE ATTRIBUTES (anno/list_attr_cloth.txt=attribute_name  attribute_type)

def _read_table_txt(path: str, ncols: int):
    """
    DeepFashion txt suele tener:
    - primera línea: header
    - segunda línea: número de líneas
    - luego filas con columnas separadas por espacios
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    data = []
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < ncols:
            continue
        data.append(parts[:ncols])
    return data


def load_category_mapping():
    # list_category_img: <img_path> <category_id>
    data = _read_table_txt(LIST_CATEGORY_IMG, ncols=2)
    df = pd.DataFrame(data, columns=["rel_path", "category_id"])
    df["category_id"] = df["category_id"].astype(int) - 1
    df["image_path"] = df["rel_path"].apply(lambda p: os.path.join(ROOT_IMG, p))
    return df


def load_partitions():
    # list_eval_partition: <img_path> <train|val|test>
    data = _read_table_txt(LIST_EVAL_PARTITION, ncols=2)
    return pd.DataFrame(data, columns=["rel_path", "split"])

def load_category_names_optional():
    if not os.path.exists(LIST_CATEGORY_CLOTH):
        return None
    # list_category_cloth: <category_name> <category_type>  (varía por versión)
    # Muchas versiones tienen 2 columnas: name + type.
    # Queremos un índice -> nombre en orden de archivo (1-indexed).
    data = _read_table_txt(LIST_CATEGORY_CLOTH, ncols=2)
    names = [row[0] for row in data]
    # Devolver dict id0->name
    return {i: names[i] for i in range(len(names))}


# -------- NUEVO: CARGA DE ATTRIBUTES --------
def load_attribute_names():
    data = _read_table_txt(LIST_ATTR_CLOTH, ncols=2)
    names = [row[0] for row in data]
    return names


def load_image_attributes(attr_names):
    n_attr = len(attr_names)
    with open(LIST_ATTR_IMG, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()][2:]

    rows = []
    for ln in lines:
        parts = ln.split()
        rel_path = parts[0]
        vals = [1 if int(v) == 1 else 0 for v in parts[1:n_attr+1]]
        row = {"rel_path": rel_path}
        for i, name in enumerate(attr_names):
            row[f"attr_{i}"] = vals[i]
        rows.append(row)

    return pd.DataFrame(rows)
# -------------------------------------------


def main():
    cat = load_category_mapping()
    part = load_partitions()
    merged = cat.merge(part, on="rel_path", how="inner")

    # Attributes
    attr_names = load_attribute_names()
    attr_df = load_image_attributes(attr_names)
    merged = merged.merge(attr_df, on="rel_path", how="left").fillna(0)

    # Validar archivos existentes
    merged = merged[merged["image_path"].apply(os.path.exists)].copy()

    merged.rename(columns={"category_id": "label"}, inplace=True)

    """
    Se agrego reindexar lables para garantizar que las etiquetas sean consecutivas (0..N-1), ya que TensorFlow exige ese formato y evita errores como “label fuera de rango” durante el entrenamiento.
    """
    # --- REINDEXAR LABELS A [0..N-1] --- (nuevo)
    label_map = {old: new for new, old in enumerate(sorted(merged["label"].unique()))}
    merged["label"] = merged["label"].map(label_map)

    # (Opcional) agregar label_name si tenemos nombres
    name_map = load_category_names_optional()
    if name_map:
        merged["label_name"] = merged["label"].map(name_map).fillna("unknown")

    # Guardar splits
    for split_name, out_name in [("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")]:
        cols = ["image_path", "label"] + [c for c in merged.columns if c.startswith("attr_")]
        if "label_name" in merged.columns:
            cols.append("label_name")
        df_split = merged[merged["split"] == split_name][cols].reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, out_name)
        df_split.to_csv(out_path, index=False)
        print("✅", out_path, "rows=", len(df_split))


if __name__ == "__main__":
    main()
