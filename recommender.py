"""
recommender.py

Loads the pre-built TF-IDF artifacts and exposes JobRecommender.recommend().
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
        self.skill_options = self._build_skill_options()

    def _build_skill_options(self, max_options: int = 1500) -> list[str]:
        """Skill choices taken from the model vocabulary (lowercase),
        ordered by how many jobs mention them (most common first)."""
        vocab = self.vectorizer.vocabulary_
        doc_freq = np.asarray((self.tfidf_matrix > 0).sum(axis=0)).ravel()
        terms = [
            (term, doc_freq[idx]) for term, idx in vocab.items()
            if not term.replace(" ", "").isdigit() and len(term) > 1
        ]
        terms.sort(key=lambda t: -t[1])
        return [t for t, _ in terms[:max_options]]

    def recommend(
        self,
        user_skills: str = "",
        user_experience: float | None = None,
        user_expected_salary: float | None = None,
        title_query: str = "",
        top_n: int = 10,
        salary_tolerance: float = 0.15,
    ):
        """Return (disclosed_df, undisclosed_df) top matches for the query."""
        df = self.df

        # Rank by skills; if none chosen, fall back to the title text.
        query_text = (user_skills or title_query).lower()
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        mask = np.ones(len(df), dtype=bool)

        # Job-title text search: every typed word must appear in the title
        for word in title_query.lower().split():
            mask &= df["title"].str.lower().str.contains(
                word, regex=False, na=False
            ).to_numpy()

        if user_experience is not None:
            mask &= ((df["minimumExperience"] <= user_experience) &
                     (df["maximumExperience"] >= user_experience)).to_numpy()

        disclosed_mask = mask & (df["salary_disclosed"] == True).to_numpy()    # noqa: E712
        undisclosed_mask = mask & (df["salary_disclosed"] == False).to_numpy()  # noqa: E712

        if user_expected_salary is not None:
            disclosed_mask &= (
                df["maximumSalary"] >= user_expected_salary * (1 - salary_tolerance)
            ).to_numpy()

        def build_result(m, cols):
            sims_filtered = np.where(m, sims, -1)
            top_idx = np.argsort(sims_filtered)[::-1][:top_n]
            top_idx = [i for i in top_idx if sims_filtered[i] > -1]
            res = df.iloc[top_idx][cols].copy()
            res["match_score"] = sims_filtered[top_idx]
            return res.reset_index(drop=True)

        return (
            build_result(disclosed_mask, DISCLOSED_COLS),
            build_result(undisclosed_mask, UNDISCLOSED_COLS),
        )
