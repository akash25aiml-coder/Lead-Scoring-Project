"""
Data cleaning for Lead Scoring project.
Drops are justified by EDA in notebooks/01_eda — not arbitrary.
"""
import pandas as pd

LEAKAGE_COLS = [
    "Tags",            # sales-rep post-hoc status note; near-perfect separator
    "Lead Quality",    # only assigned to already-engaged leads (56.7% vs 21.5% conv.)
    "Lead Profile",    # same failure mode (48.9% vs 13.7% conv.)
]

ZERO_VARIANCE_COLS = [
    "Magazine", "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content", "Get updates on DM Content",
    "I agree to pay the amount through cheque", "Search",
    "Newspaper Article", "X Education Forums", "Newspaper",
    "Digital Advertisement", "Through Recommendations", "Do Not Call",
]

ID_COLS = ["Prospect ID", "Lead Number"]  # identifiers, not features

# 78.5% missing after "Select" normalization — too sparse to impute honestly
SPARSE_DROP_COLS = ["How did you hear about X Education"]

# High-cardinality categoricals that get one-hot encoded downstream.
# Rare levels (<1% frequency) collapsed into "Other" — flag #2 fix:
# prevents RF/XGBoost from overfitting on categories with 1-2 samples.
HIGH_CARD_COLS = ["Country", "Specialization", "Last Activity", "Last Notable Activity", "City"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace("Select", pd.NA)
    df = df.drop(
        columns=LEAKAGE_COLS + ZERO_VARIANCE_COLS + SPARSE_DROP_COLS,
        errors="ignore",
    )
    return df


def group_rare_categories(df: pd.DataFrame, freq_maps: dict, cols=HIGH_CARD_COLS, threshold=0.01) -> pd.DataFrame:
    """Collapse levels below `threshold` frequency into 'Other', using
    frequencies computed on TRAIN ONLY (freq_maps) to avoid leakage into
    val/test."""
    df = df.copy()
    for col in cols:
        if col not in df.columns or col not in freq_maps:
            continue
        rare = freq_maps[col]
        df[col] = df[col].apply(lambda x: "Other" if x in rare else x)
    return df


def fit_rare_category_map(df_train: pd.DataFrame, cols=HIGH_CARD_COLS, threshold=0.01) -> dict:
    """Compute rare-category sets from TRAINING data only."""
    freq_maps = {}
    for col in cols:
        if col not in df_train.columns:
            continue
        freq = df_train[col].value_counts(normalize=True, dropna=True)
        freq_maps[col] = set(freq[freq < threshold].index)
    return freq_maps


def split_features(df: pd.DataFrame):
    y = df["Converted"]
    X = df.drop(columns=["Converted"] + ID_COLS, errors="ignore")
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num_cols, cat_cols


if __name__ == "__main__":
    df = load_and_clean("data/Leads.csv")
    print("Shape after drop:", df.shape)
