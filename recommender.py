"""
recommender.py

Loads the pre-built artifacts (JSON + .npz + CSV, no pickle) and exposes
JobRecommender.recommend(). Kept independent of Streamlit so it can be
reused or tested on its own.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

DISCLOSED_COLS = [
    "title", "companyName", "minimumSalary", "maximumSalary",
    "minimumExperience", "maximumExperience",
]
UNDISCLOSED_COLS = [
    "title", "companyName", "minimumExperience", "maximumExperience",
]

# Score weights
W_COVERAGE = 0.7      # share of chosen skills a job lists
W_SKILL_COS = 0.3     # IDF-weighted skill similarity (rare skills count more)
W_SKILLS_WITH_TITLE = 0.85   # when both skills and a title query are given
W_TITLE = 0.15


class JobRecommender:
    """Skill matrix + title TF-IDF + job dataframe."""

    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        d = Path(artifacts_dir)

        # --- skills ---
        sk = json.loads((d / "skills_vocab.json").read_text())
        self.skills: list[str] = sk["skills"]
        self.skill_idf = np.asarray(sk["idf"], dtype=np.float32)
        self.skill_index = {s: i for i, s in enumerate(self.skills)}
        self.skill_matrix = sp.load_npz(d / "skill_matrix.npz").tocsr().astype(np.float32)
        self.skill_csc = self.skill_matrix.tocsc()          # fast column access
        self.skill_tfidf = normalize(self.skill_matrix @ sp.diags(self.skill_idf)).tocsr()
        self.skill_doc_freq = np.asarray(self.skill_matrix.sum(axis=0)).ravel()

        # --- title ---
        t = json.loads((d / "title_vocab.json").read_text())
        self.title_idf = np.asarray(t["idf"], dtype=np.float32)
        self.title_counter = CountVectorizer(
            vocabulary=t["vocabulary"],
            ngram_range=tuple(t["ngram_range"]),
            stop_words="english",
        )
        self.title_tfidf = sp.load_npz(d / "title_tfidf.npz").tocsr()

        # --- jobs ---
        self.df = pd.read_csv(d / "jobs_model_ready.csv")
        self.title_lower = self.df["title"].fillna("").astype(str).str.lower()
        self.salary_disclosed = self.df["salary_disclosed"].fillna(False).astype(bool).to_numpy()

        if not (len(self.df) == self.skill_matrix.shape[0] == self.title_tfidf.shape[0]):
            raise ValueError(
                "Artifacts are out of sync (row counts differ). Re-run train_model.py."
            )

        self.skill_options = self._build_skill_options()

    def _build_skill_options(self, max_options: int = 4000) -> list[str]:
        """Real skills only, most common first."""
        order = np.argsort(-self.skill_doc_freq, kind="stable")[:max_options]
        return [self.skills[i] for i in order]

    # ------------------------------------------------------------ scoring
    def _title_similarity(self, query: str) -> np.ndarray:
        counts = self.title_counter.transform([query.lower()])
        vec = normalize(counts @ sp.diags(self.title_idf))
        return np.asarray((self.title_tfidf @ vec.T).todense()).ravel()

    @staticmethod
    def _relative(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Scale so the best candidate in the current search equals 1."""
        best = x[mask].max() if mask.any() else 0
        return x / best if best > 0 else x

    # ---------------------------------------------------------- recommend
    def recommend(
        self,
        selected_skills: list[str] | None = None,
        user_experience: float | None = None,
        user_expected_salary: float | None = None,
        title_query: str = "",
        top_n: int = 10,
        salary_tolerance: float = 0.15,
    ):
        """Return (disclosed_df, undisclosed_df) top matches for the query."""
        df = self.df
        n = len(df)
        title_query = (title_query or "").strip()

        cols = [self.skill_index[s.lower()] for s in (selected_skills or [])
                if s.lower() in self.skill_index]
        cols = list(dict.fromkeys(cols))

        # ---- filters ----
        mask = np.ones(n, dtype=bool)
        for word in title_query.lower().split():
            mask &= self.title_lower.str.contains(word, regex=False).to_numpy()

        if user_experience is not None:
            mask &= ((df["minimumExperience"] <= user_experience) &
                     (df["maximumExperience"] >= user_experience)).to_numpy()

        disclosed_mask = mask & self.salary_disclosed
        undisclosed_mask = mask & ~self.salary_disclosed

        if user_expected_salary is not None:
            disclosed_mask &= (
                df["maximumSalary"] >= user_expected_salary * (1 - salary_tolerance)
            ).to_numpy()

        # ---- score ----
        score = np.zeros(n)
        title_sim = self._title_similarity(title_query) if title_query else None

        if cols:
            coverage = np.asarray(self.skill_csc[:, cols].mean(axis=1)).ravel()
            q = np.zeros(len(self.skills), dtype=np.float32)
            q[cols] = self.skill_idf[cols]
            q /= np.linalg.norm(q)
            cos = np.asarray(self.skill_tfidf @ q).ravel()
            score = W_COVERAGE * coverage + W_SKILL_COS * self._relative(cos, mask)
            if title_sim is not None:
                score = (W_SKILLS_WITH_TITLE * score
                         + W_TITLE * self._relative(title_sim, mask))
        elif title_sim is not None:
            score = self._relative(title_sim, mask)

        def build_result(m, out_cols):
            idx = np.flatnonzero(m & (score > 0))      # only real matches
            if idx.size == 0:
                return pd.DataFrame(columns=out_cols + ["match_score"])
            if idx.size > top_n:
                idx = idx[np.argpartition(-score[idx], top_n - 1)[:top_n]]
            idx = idx[np.argsort(-score[idx], kind="stable")]
            res = df.iloc[idx][out_cols].copy()
            res["match_score"] = score[idx]
            return res.reset_index(drop=True)

        return (
            build_result(disclosed_mask, DISCLOSED_COLS),
            build_result(undisclosed_mask, UNDISCLOSED_COLS),
        )