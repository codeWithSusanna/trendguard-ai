"""
Cleans four medical CSV datasets (breast cancer, dengue, thyroid, asthma/symptom).
Originals are read-only inputs; cleaned copies are written to OUTPUT_DIR.

Run: python3 clean_data.py
"""
import pandas as pd

IN_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/mnt/user-data/outputs"


def report(name, before, after, missing_before, missing_after, dup_before, dup_after, class_col=None):
    print(f"\n=== {name} ===")
    print(f"Shape: {before} -> {after}")
    print(f"Missing values (total): {missing_before} -> {missing_after}")
    print(f"Duplicate rows: {dup_before} -> {dup_after}")
    if class_col is not None:
        print(f"Class distribution ({class_col.name}):")
        print(class_col.to_string())


# ---------------------------------------------------------------------------
# 1. CANCER dataset
# ---------------------------------------------------------------------------
cancer = pd.read_csv(f"{IN_DIR}/Cancer_Data.csv")
before_shape, before_na, before_dup = cancer.shape, cancer.isna().sum().sum(), cancer.duplicated().sum()

# 'Unnamed: 32' is 100% empty (artifact of a trailing comma in the source file) -> drop.
# 'id' is a unique identifier, not a feature -> kept for traceability, not used as a feature.
cancer_clean = cancer.drop(columns=["Unnamed: 32"])
cancer_clean = cancer_clean.drop_duplicates()  # none expected, safety check
assert cancer_clean["id"].is_unique, "Duplicate patient IDs found in cancer data"

after_shape = cancer_clean.shape
after_na = cancer_clean.isna().sum().sum()
after_dup = cancer_clean.duplicated().sum()
report("Cancer_Data.csv", before_shape, after_shape, before_na, after_na, before_dup, after_dup,
       cancer_clean["diagnosis"].value_counts())
cancer_clean.to_csv(f"{OUT_DIR}/cancer_cleaned.csv", index=False)


# ---------------------------------------------------------------------------
# 2. DENGUE dataset
# ---------------------------------------------------------------------------
dengue = pd.read_csv(f"{IN_DIR}/Dengue_diseases_dataset_modified__1_.csv")
before_shape, before_na, before_dup = dengue.shape, dengue.isna().sum().sum(), dengue.duplicated().sum()

dengue_clean = dengue.drop_duplicates().copy()

# Missing lab values (wbc_count, platelet_count, platelet_distribution_width) are <2.5% of
# rows each and numeric/skewed -> median imputation (per dengue_label group) preserves the
# distribution without discarding otherwise-valid patient records or inventing extreme values.
lab_cols = ["wbc_count", "platelet_count", "platelet_distribution_width"]
for col in lab_cols:
    dengue_clean[col] = dengue_clean.groupby("dengue_label")[col].transform(lambda s: s.fillna(s.median()))

# 'gender' has a valid third category 'Child' (pediatric patients) in addition to Male/Female -> kept as-is.
# 'rbc_count' and 'differential_count' are stored as 0/1 flags rather than the continuous
# measurements described in the data dictionary. This looks like an upstream encoding issue,
# but the values are internally consistent (no NaNs/negatives), so they are left untouched
# rather than guessed/rewritten, and flagged in this report instead.

after_shape = dengue_clean.shape
after_na = dengue_clean.isna().sum().sum()
after_dup = dengue_clean.duplicated().sum()
report("Dengue_diseases_dataset_modified__1_.csv", before_shape, after_shape, before_na, after_na,
       before_dup, after_dup, dengue_clean["dengue_label"].value_counts())
print("NOTE: rbc_count/differential_count are binary (0/1) in the source file, not counts as")
print("      described in the data dictionary. Left as-is (not guessed) - flag for data owner.")
dengue_clean.to_csv(f"{OUT_DIR}/dengue_cleaned.csv", index=False)


# ---------------------------------------------------------------------------
# 3. THYROID dataset
# ---------------------------------------------------------------------------
thyroid = pd.read_csv(f"{IN_DIR}/Thyroid-Dataset.csv")
before_shape, before_na, before_dup = thyroid.shape, thyroid.isna().sum().sum(), thyroid.duplicated().sum()

thyroid_clean = thyroid.drop_duplicates().copy()  # 102 exact duplicate patient rows

# A handful of 'age' values are biologically impossible (e.g. 455, 65511, 65526 - integer
# overflow/typo artifacts), unlike the rest of the column (1-97, all plausible). These rows
# are removed as corrupted records rather than guessed at; every other row is preserved.
invalid_age = thyroid_clean["age"] > 120
n_invalid_age = invalid_age.sum()
thyroid_clean = thyroid_clean[~invalid_age]

# 'sex' and the lab values (TSH, T3, TT4, T4U, FTI) are left as NaN when missing: these are
# clinical measurements/attributes that cannot be reliably inferred, and imputing them would
# mean fabricating patient data. Missingness is preserved rather than guessed or dropped.

after_shape = thyroid_clean.shape
after_na = thyroid_clean.isna().sum().sum()
after_dup = thyroid_clean.duplicated().sum()
report("Thyroid-Dataset.csv", before_shape, after_shape, before_na, after_na, before_dup, after_dup,
       thyroid_clean["class"].value_counts())
print(f"NOTE: removed {n_invalid_age} rows with impossible age values (>120 years).")
print("Missing counts kept (not imputed):")
print(thyroid_clean.isna().sum()[thyroid_clean.isna().sum() > 0].to_string())
thyroid_clean.to_csv(f"{OUT_DIR}/thyroid_cleaned.csv", index=False)


# ---------------------------------------------------------------------------
# 4. ASTHMA / symptom-severity dataset (processed-data.csv)
# ---------------------------------------------------------------------------
asthma = pd.read_csv(f"{IN_DIR}/processed-data.csv")
before_shape, before_na, before_dup = asthma.shape, asthma.isna().sum().sum(), asthma.duplicated().sum()

# This is a fully binary/one-hot encoded symptom-severity dataset with only 5,760 distinct
# symptom combinations, each repeated exactly 55 times in the raw file. The repeats carry no
# extra information (same feature vector -> same label every time) and would bias any model
# trained on it toward these exact duplicate rows, so exact duplicates are dropped, keeping
# one row per unique combination.
asthma_clean = asthma.drop_duplicates().copy()

after_shape = asthma_clean.shape
after_na = asthma_clean.isna().sum().sum()
after_dup = asthma_clean.duplicated().sum()
sev_cols = ["Severity_None", "Severity_Mild", "Severity_Moderate"]
sev_dist = asthma_clean[sev_cols].sum()
sev_dist.name = "count"
report("processed-data.csv (asthma/symptom severity)", before_shape, after_shape, before_na, after_na,
       before_dup, after_dup, sev_dist)
asthma_clean.to_csv(f"{OUT_DIR}/asthma_cleaned.csv", index=False)

print("\nAll cleaned files written to", OUT_DIR)
