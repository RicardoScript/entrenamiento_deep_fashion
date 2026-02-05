import json
import pandas as pd

TRAIN_CSV = "data/processed/train.csv"
OUT = "data/processed/labels.json"

# Si tu CSV no tiene label_name, puedes crear nombres dummy: "class_0", etc.
def main():
    df = pd.read_csv(TRAIN_CSV)

    if "label_name" in df.columns:
        mapping = (
            df[["label", "label_name"]]
            .drop_duplicates()
            .sort_values("label")
        )
        labels = mapping["label_name"].astype(str).tolist()
    else:
        n = int(df["label"].nunique())
        labels = [f"class_{i}" for i in range(n)]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("✅ labels.json generado:", OUT, "(", len(labels), "clases )")

if __name__ == "__main__":
    main()
