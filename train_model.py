import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "job_cleaned.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MIN_SKILL_DF = 5       # a skill must appear in at least this many listings
MIN_TITLE_DF = 2
TITLE_NGRAMS = (1, 2)
TITLE_MAX_FEATURES = 20000

# Columns the app needs at runtime (missing ones are skipped)
KEEP_COLS = [
    "title", "companyName", "minimumSalary", "maximumSalary",
    "minimumExperience", "maximumExperience", "salary_disclosed",
    "location_clean",
]


def parse_skills(text) -> list[str]:
    """'Machine Learning, SQL , sql' -> ['machine learning', 'sql']"""
    if not isinstance(text, str):
        return []
    skills = dict.fromkeys(s.strip().lower() for s in text.split(","))
    return [s for s in skills if s]


def train(data_path=DATA_PATH, artifacts_dir=ARTIFACTS_DIR):
    data_path, out = Path(data_path), Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    df["title"] = df["title"].fillna("").astype(str)
    df["finalSkill"] = df["finalSkill"].fillna("")
    n = len(df)

    # ---- skills: one token per comma-separated skill, binary ----
    skill_cv = CountVectorizer(
        analyzer=parse_skills, binary=True, min_df=MIN_SKILL_DF, dtype=np.float32
    )
    skill_matrix = skill_cv.fit_transform(df["finalSkill"]).tocsr()
    skills = [str(s) for s in skill_cv.get_feature_names_out()]
    doc_freq = np.asarray(skill_matrix.sum(axis=0)).ravel()
    skill_idf = np.log((1 + n) / (1 + doc_freq)) + 1

    # ---- title: separate TF-IDF ----
    title_vec = TfidfVectorizer(
        stop_words="english", ngram_range=TITLE_NGRAMS,
        min_df=MIN_TITLE_DF, max_features=TITLE_MAX_FEATURES,
    )
    title_tfidf = title_vec.fit_transform(df["title"]).tocsr()

    print("Rows:", n)
    print("Skill matrix shape:", skill_matrix.shape)
    print("Title TF-IDF shape:", title_tfidf.shape)

    # ---- save (JSON + npz, no pickle) ----
    (out / "skills_vocab.json").write_text(
        json.dumps({"skills": skills, "idf": [float(x) for x in skill_idf]})
    )
    sp.save_npz(out / "skill_matrix.npz", skill_matrix)

    (out / "title_vocab.json").write_text(json.dumps({
        "vocabulary": {t: int(i) for t, i in title_vec.vocabulary_.items()},
        "idf": [float(x) for x in title_vec.idf_],
        "ngram_range": list(TITLE_NGRAMS),
    }))
    sp.save_npz(out / "title_tfidf.npz", title_tfidf)

    df[[c for c in KEEP_COLS if c in df.columns]].to_csv(
        out / "jobs_model_ready.csv", index=False
    )
    print(f"Artifacts saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--out", default=ARTIFACTS_DIR)
    args = parser.parse_args()
    train(args.data, args.out)