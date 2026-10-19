# 💼 Job Recommender

A content-based job recommendation system built on ~97,000 Indian job listings. Pick your skills, optionally search by job title, add your experience and expected salary, and get the closest-matching openings, ranked by how well they fit.

**🚀 Live app:** https://job-recommender-web.streamlit.app/

---

## Features

- **Skill picker** – choose from a searchable dropdown instead of typing a whole skill list. Start typing to filter, and pick as many skills as you like. Options are shown capitalized (e.g. *Machine Learning*, *SQL*).
- **Job title search** – an optional text box that keeps only jobs whose title contains every word you type.
- **Experience and salary filters** – leave either at `0` to skip it.
- **Two result tables** – jobs *with* salary info and jobs *without*, each ranked by match score.
- **Readable results** – compact columns that fit on screen: salary shown as a range in LPA (e.g. `₹5.4 – 7.2 LPA`), experience as `5 – 10 yrs`, and match shown as a percentage bar.
- **Polished UI** – dark theme with a glass-style form card, a slowly drifting aurora background and a soft cursor glow. Motion is disabled automatically for users who prefer reduced motion.

## How it works

1. **Data cleaning** – Raw listings are cleaned in `Notebooks/Job_Recommender_Cleaning.ipynb`: skills are extracted from free-text descriptions using an **Aho-Corasick automaton**, and inconsistent location strings are normalized with **fuzzy matching**.
2. **Feature building** – `train_model.py` combines each listing's title and extracted skills into one text field (skills are repeated to weight them), then fits a **TF-IDF vectorizer** (`scikit-learn`, unigrams + bigrams, 5,000 max features, `min_df=2`) over the corpus.
3. **Recommendation** – `recommender.py` filters listings, then scores and ranks them (see below).
4. **Interface** – `app.py` is a Streamlit front-end that loads the pre-built artifacts once (`st.cache_resource`) and renders the form and result tables.

### Filtering

| Filter | Rule |
|---|---|
| Job title | every typed word must appear in the title |
| Experience | `minimumExperience ≤ your years ≤ maximumExperience` |
| Salary | disclosed-salary jobs only: `maximumSalary ≥ expected salary × 0.85` (±15% tolerance) |

### Match score

Cosine similarity alone gives low numbers for short queries, because a two-skill query is compared against listings with dozens of terms. So the score blends two signals:

```
match = 0.7 × skill coverage + 0.3 × relative cosine similarity
```

- **Skill coverage** – the share of your selected skills that the listing actually mentions (0 to 1).
- **Relative cosine similarity** – the TF-IDF cosine score, scaled so the best result in the current search equals 1. It breaks ties between listings that cover the same skills.

If you search by title only (no skills), ranking falls back to the title's TF-IDF similarity.

## Run it locally

```bash
git clone https://github.com/Moyukh-Mondal/Job_Recommender.git
cd Job_Recommender
pip install -r requirements.txt
streamlit run app.py
```

The app loads pre-built model artifacts from `artifacts/` (`tfidf_vectorizer.pkl`, `tfidf_matrix.npz`, `jobs_model_ready.csv`). These are included in the repo via Git LFS, so no training step is needed to run the app. If the files look like tiny text pointers, run `git lfs pull`.

### Regenerating the model artifacts

If `data/job_cleaned.csv` is updated, rebuild the artifacts with:

```bash
python train_model.py
```

This reads `data/job_cleaned.csv` and writes a fresh vectorizer, TF-IDF matrix and model-ready CSV to `artifacts/`. The skill dropdown is built from the new vectorizer's vocabulary automatically.

## Project structure

```
├── app.py                          # Streamlit UI (form, styling, results)
├── recommender.py                  # JobRecommender class (loads artifacts, filters, scores)
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

- **Python** – pandas, NumPy
- **scikit-learn** – TF-IDF vectorization, cosine similarity
- **SciPy** – sparse matrix storage for the TF-IDF matrix
- **Streamlit** – web interface and deployment (with custom CSS/JS for the theme)
- **Aho-Corasick** – fast multi-pattern skill extraction from raw text
- **Fuzzy matching** – location string normalization

## Notes and limitations

- Large data/artifact files (`.csv`, `.xlsx`, `.npz`) are tracked with **Git LFS**.
- `recommender.py` is kept independent of Streamlit so it can be reused or tested standalone.
- The skill dropdown is built from the model vocabulary (top 1,500 terms by number of listings), which is made from titles *and* skills. A few title words such as "engineer" can therefore appear as options.
- Streamlit caches the recommender object. After editing `recommender.py`, restart the app or bump the `version` argument of `load_recommender()` in `app.py`.
- Salary values are shown as stored in the dataset; listings with `0` salary are displayed as "Not specified".

## Ideas for future work

- Build the skill list only from the cleaned `finalSkill` column.
- Add location and job-type filters.
- Try embedding-based similarity (e.g. sentence transformers) alongside TF-IDF.
- Add a "why this match" view showing which of your skills each listing covers.
