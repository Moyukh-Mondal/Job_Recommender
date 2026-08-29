# 💼 Job Recommender

A content-based job recommendation system built on ~97,000 Indian job listings. Enter your skills, experience, and expected salary — get back the closest-matching openings, ranked by similarity.

**🚀 Live app:** https://job-recommender-web.streamlit.app/

---

## How it works

1. **Data cleaning** — Raw listings are cleaned in `Job_Recommender_Cleaning.ipynb`: skills are extracted from free-text descriptions using an **Aho-Corasick automaton**, and inconsistent location strings are normalized with **fuzzy matching**.
2. **Feature building** — `train_model.py` combines each listing's title and extracted skills into a single text field, then fits a **TF-IDF vectorizer** (`scikit-learn`, unigrams + bigrams, 5,000 max features) over the corpus.
3. **Recommendation** — `recommender.py` embeds a user's typed skills with the same vectorizer and ranks listings by **cosine similarity**, optionally filtered by years of experience and expected salary (±15% tolerance).
4. **Interface** — `app.py` is a Streamlit front-end: a simple form (skills, experience, expected salary) returns two ranked tables — jobs with disclosed salary info and jobs without — each showing a match score.

## Run it locally

```bash
git clone https://github.com/Moyukh-Mondal/Job_Recommender.git
cd Job_Recommender
pip install -r requirements.txt
streamlit run app.py
```

The app loads pre-built model artifacts from `artifacts/` (`tfidf_vectorizer.pkl`, `tfidf_matrix.npz`, `jobs_model_ready.csv`) — these are already included in the repo via Git LFS, so no training step is required to run the app.

### Regenerating the model artifacts

If `data/job_cleaned.csv` is updated, rebuild the artifacts with:

```bash
python train_model.py
```

This reads `data/job_cleaned.csv` and writes a fresh vectorizer, TF-IDF matrix, and model-ready CSV to `artifacts/`.

## Project structure

```
├── app.py                          # Streamlit UI
├── recommender.py                  # JobRecommender class (loads artifacts, computes similarity)
├── train_model.py                  # Builds TF-IDF artifacts from cleaned data
├── requirements.txt
├── artifacts/                      # Pre-built model artifacts (Git LFS)
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.npz
│   └── jobs_model_ready.csv
├── data/                           # Raw and cleaned datasets (Git LFS)
│   ├── indian-job-market-dataset-2025.xlsx
│   └── job_cleaned.csv
└── Notebooks/
    └── Job_Recommender_Cleaning.ipynb   # Data cleaning & feature extraction
```

## Tech stack

- **Python** — pandas, NumPy
- **scikit-learn** — TF-IDF vectorization, cosine similarity
- **SciPy** — sparse matrix storage for the TF-IDF matrix
- **Streamlit** — web interface & deployment
- **Aho-Corasick** — fast multi-pattern skill extraction from raw text
- **Fuzzy matching** — location string normalization

## Notes

- Large data/artifact files (`.csv`, `.xlsx`, `.npz`) are tracked with **Git LFS**.
- `recommender.py` is kept independent of Streamlit so it can be reused or tested standalone.
