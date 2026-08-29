"""
recommender.py

Loads the pre-built TF-IDF artifacts and exposes recommend_jobs().
Kept independent of Streamlit so it can be reused or tested on its own.
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = "artifacts"

DISCLOSED_COLS = [
    "title", "companyName", "minimumSalary", "maximumSalary",
    "minimumExperience", "maximumExperience",
]
UNDISCLOSED_COLS = [
    "title", "companyName", "minimumExperience", "maximumExperience",
]


class JobRecommender:
    """Wraps the fitted vectorizer + TF-IDF matrix + job dataframe."""

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        with open(f"{artifacts_dir}/tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        self.tfidf_matrix = sp.load_npz(f"{artifacts_dir}/tfidf_matrix.npz")
        self.df = pd.read_csv(f"{artifacts_dir}/jobs_model_ready.csv")

    def recommend(
        self,
        user_skills: str,
        user_experience: float | None = None,
        user_expected_salary: float | None = None,
        top_n: int = 10,
        salary_tolerance: float = 0.15,
    ):
        """Return (disclosed_df, undisclosed_df) top matches for the query."""
        df = self.df

        query_vec = self.vectorizer.transform([user_skills.lower()])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        exp_mask = np.ones(len(df), dtype=bool)
        if user_experience is not None:
            exp_mask = (df["minimumExperience"] <= user_experience) & \
                       (df["maximumExperience"] >= user_experience)

        disclosed_mask = exp_mask & (df["salary_disclosed"] == True)   # noqa: E712
        undisclosed_mask = exp_mask & (df["salary_disclosed"] == False)  # noqa: E712

        if user_expected_salary is not None:
            disclosed_mask &= (
                df["maximumSalary"] >= user_expected_salary * (1 - salary_tolerance)
            )

        def build_result(mask, cols):
            sims_filtered = np.where(mask, sims, -1)
            top_idx = np.argsort(sims_filtered)[::-1][:top_n]
            top_idx = [i for i in top_idx if sims_filtered[i] > -1]
            res = df.iloc[top_idx][cols].copy()
            res["match_score"] = sims_filtered[top_idx]
            return res.reset_index(drop=True)

        disclosed_results = build_result(disclosed_mask, DISCLOSED_COLS)
        undisclosed_results = build_result(undisclosed_mask, UNDISCLOSED_COLS)

        return disclosed_results, undisclosed_results
