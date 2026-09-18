"""
clean_data.py
Cleans three medical datasets: Kidney Disease, Stroke, and Indian Liver Patient (ILPD).

Inputs (read-only, never modified):
    /mnt/user-data/uploads/kidney_disease.csv
    /mnt/user-data/uploads/healthcare-dataset-stroke-data.csv
    /mnt/user-data/uploads/Indian_Liver_Patient_Dataset__ILPD_.csv

Outputs:
    /mnt/user-data/outputs/kidney_disease_cleaned.csv
    /mnt/user-data/outputs/stroke_data_cleaned.csv
    /mnt/user-data/outputs/liver_patient_cleaned.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

IN_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/mnt/user-data/outputs"


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =========================================================================
# 1. KIDNEY DISEASE
# =========================================================================
def clean_kidney():
    banner("KIDNEY DISEASE DATASET")
    df = pd.read_csv(f"{IN_DIR}/kidney_disease.csv")

    print(f"BEFORE  shape={df.shape}  missing_total={df.isnull().sum().sum()}  "
          f"duplicates={df.duplicated().sum()}")
    print("Class distribution BEFORE (raw, uncleaned labels):")
    print(df["classification"].value_counts(dropna=False).to_string())

    cat_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
                "classification"]
    numeric_as_text_cols = ["pcv", "wc", "rc"]           # numeric cols stored as text
    numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo"]

    # --- Fix inconsistent values: strip stray whitespace/tabs, normalize '?' -> NaN ---
    for c in cat_cols + numeric_as_text_cols:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"?": pd.NA, "": pd.NA})

    # pcv/wc/rc are numeric but were read as text because of stray '?'/tabs
    for c in numeric_as_text_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        numeric_cols.append(c)

    # --- Duplicates: drop exact duplicate patient records (ignore id) ---
    dup_mask = df.drop(columns=["id"]).duplicated()
    df = df.loc[~dup_mask].reset_index(drop=True)

    # --- Missing values: median for numeric (robust to outliers), mode for categorical ---
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

    # --- Encoding: map clean binary/categorical text to 0/1 ---
    enc_maps = {
        "rbc": {"normal": 1, "abnormal": 0},
        "pc": {"normal": 1, "abnormal": 0},
        "pcc": {"present": 1, "notpresent": 0},
        "ba": {"present": 1, "notpresent": 0},
        "htn": {"yes": 1, "no": 0},
        "dm": {"yes": 1, "no": 0},
        "cad": {"yes": 1, "no": 0},
        "appet": {"good": 1, "poor": 0},
        "pe": {"yes": 1, "no": 0},
        "ane": {"yes": 1, "no": 0},
        "classification": {"ckd": 1, "notckd": 0},
    }
    for c, m in enc_maps.items():
        df[c] = df[c].map(m).astype(int)

    # --- Data types: whole-count measurements -> int, decimal measurements -> float ---
    int_cols = ["age", "bp", "al", "su", "bgr", "pcv", "wc"]
    float_cols = ["sg", "bu", "sc", "sod", "pot", "hemo", "rc"]
    for c in int_cols:
        df[c] = df[c].round().astype(int)
    for c in float_cols:
        df[c] = df[c].astype(float)

    print(f"\nAFTER   shape={df.shape}  missing_total={df.isnull().sum().sum()}  "
          f"duplicates={df.drop(columns=['id']).duplicated().sum()}")
    print("Class distribution AFTER (1=ckd, 0=notckd):")
    print(df["classification"].value_counts().to_string())

    out_path = f"{OUT_DIR}/kidney_disease_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


# =========================================================================
# 2. STROKE
# =========================================================================
def clean_stroke():
    banner("STROKE DATASET")
    df = pd.read_csv(f"{IN_DIR}/healthcare-dataset-stroke-data.csv")

    print(f"BEFORE  shape={df.shape}  missing_total={df.isnull().sum().sum()}  "
          f"duplicates={df.duplicated().sum()}")
    print("Missing by column (BEFORE):")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print("Class distribution BEFORE:")
    print(df["stroke"].value_counts().to_string())

    # --- Duplicates ---
    df = df.drop_duplicates().reset_index(drop=True)

    # --- Missing values: bmi is the only column with nulls -> median imputation.
    # age is intentionally left as float: this dataset records fractional ages
    # (e.g. 0.08, 1.32) for infants/toddlers, so rounding would corrupt valid data.
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    print(f"\nAFTER   shape={df.shape}  missing_total={df.isnull().sum().sum()}  "
          f"duplicates={df.duplicated().sum()}")

    out_path = f"{OUT_DIR}/stroke_data_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    # --- Class imbalance: demonstrate correct, leakage-free handling ---
    # Split BEFORE any resampling; resample ONLY the training partition.
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("\nClass imbalance handling (demonstration, not baked into the saved CSV):")
    print("  Train BEFORE resampling:", y_train.value_counts().to_dict())
    print("  Test  (untouched)      :", y_test.value_counts().to_dict())

    train_df = pd.concat([X_train, y_train], axis=1)
    majority = train_df[train_df["stroke"] == 0]
    minority = train_df[train_df["stroke"] == 1]
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=42
    )
    train_balanced = pd.concat([majority, minority_upsampled])
    print("  Train AFTER oversampling minority class:",
          train_balanced["stroke"].value_counts().to_dict())
    print("  (Test set distribution is left unchanged to reflect real-world imbalance.)")

    return df


# =========================================================================
# 3. LIVER (ILPD)
# =========================================================================
def clean_liver():
    banner("LIVER (ILPD) DATASET")

    # Headers in the raw file are actually the first data row (corrupted header).
    # Verified column order (UCI ILPD documentation): Age, Gender, Total_Bilirubin,
    # Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,
    # Aspartate_Aminotransferase, Total_Protiens, Albumin,
    # Albumin_and_Globulin_Ratio, Dataset (1=liver patient, 2=not).
    cols = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
            "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"]
    df = pd.read_csv(f"{IN_DIR}/Indian_Liver_Patient_Dataset__ILPD_.csv",
                      header=None, names=cols)

    print(f"BEFORE  shape={df.shape}  missing_total={df.isnull().sum().sum()}  "
          f"duplicates={df.duplicated().sum()}")
    print("Class distribution BEFORE (1=liver patient, 2=not):")
    print(df["Dataset"].value_counts().to_string())

    # --- Duplicates: drop exact duplicate patient records ---
    df = df.drop_duplicates().reset_index(drop=True)

    # --- Missing values: only Albumin_and_Globulin_Ratio has nulls -> median impute ---
    df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
        df["Albumin_and_Globulin_Ratio"].median()
    )

    # --- Gender: already consistent (Male/Female); light normalization for safety ---
    df["Gender"] = df["Gender"].astype("string").str.strip().str.title()

    print(f"\nAFTER   shape={df.shape}  missing_total={df.isnull().sum().sum()}  "
          f"duplicates={df.duplicated().sum()}")
    print("Class distribution AFTER:")
    print(df["Dataset"].value_counts().to_string())

    out_path = f"{OUT_DIR}/liver_patient_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    kidney = clean_kidney()
    stroke = clean_stroke()
    liver = clean_liver()
    banner("DONE")
