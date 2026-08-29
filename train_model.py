"""
train_model.py

Run this ONCE (or whenever job_cleaned.csv is refreshed) to build the
TF-IDF model artifacts that the Streamlit app loads at runtime.

Usage:
    python train_model.py

Input:
    data/job_cleaned.csv   (output of the cleaning notebook)

Output (written to artifacts/):
    tfidf_vectorizer.pkl
    tfidf_matrix.npz
    jobs_model_ready.csv
"""

import pickle
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/job_cleaned.csv"
ARTIFACTS_DIR = "artifacts"


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["finalSkill"] = df["finalSkill"].fillna("")
    df["combined_text"] = df["title"] + " " + df["finalSkill"] + " " + df["finalSkill"]
    return df


def train():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = build_combined_text(df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    with open(f"{ARTIFACTS_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(f"{ARTIFACTS_DIR}/tfidf_matrix.npz", tfidf_matrix)
    df.to_csv(f"{ARTIFACTS_DIR}/jobs_model_ready.csv", index=False)

    print(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    train()
