# 💼 Job Recommender

A content-based job recommendation system built on Indian job listings (roughly 87,000 after cleaning and de-duplication). Pick your skills, optionally search by job title, add your experience and expected salary, and get the closest-matching openings, ranked by how well they fit.

**🚀 Live app:** https://job-recommender-web.streamlit.app/

<!-- Add a screenshot: save it as assets/screenshot.png and uncomment the next line -->
<!-- ![App screenshot](assets/screenshot.png) -->

---

## Features

- **Skill picker** – choose from a searchable dropdown instead of typing a skill list. Start typing to filter and pick as many skills as you like. Options are capitalized (e.g. *Machine Learning*, *SQL*).
- **Job title search** – an optional text box that keeps only jobs whose title contains every word you type.
- **Experience and salary filters** – leave either at `0` to skip it.
- **Two result tables** – jobs *with* salary info and jobs *without*, each ranked by match score.
- **Readable results** – compact columns that fit on screen: salary as a range in LPA (e.g. `₹5.4 – 7.2 LPA`), experience as `5 – 10 yrs`, and the match shown as a percentage bar.
- **Polished UI** – dark theme, glass-style form card, a slowly drifting aurora background and a soft cursor glow. Motion is switched off for users who prefer reduced motion.

## How it works

### 1. Data cleaning (`Notebooks/Job_Recommender_Cleaning.ipynb`)

The raw dataset (`indian-job-market-dataset-2025.xlsx`) is cleaned in a notebook:

- **Duplicates** – ID columns are dropped first, then exact duplicates are removed. After title and company cleaning, postings with the same title, company, location, salary range and experience range are de-duplicated again (the one with the most skills is kept).
- **Salary** – free-text salaries are turned into numeric minimum and maximum values. Monthly salaries (`15,000/month`) are annualised, USD salaries are converted to INR at both ends of the range, and a posting only counts as *disclosed* if its maximum salary is above 0.
- **Skills** – skills come from the structured tags, falling back to skills found in the job description with an **Aho-Corasick automaton**. Both go through the same filter: a skill must appear in at least 5 listings, and generic words, job-title words, education boilerplate and city names are removed.
- **Locations** – location strings are normalized with **fuzzy matching** (rapidfuzz) plus a hand-checked spelling dictionary.
- **Titles and companies** – recruiter noise ("urgent hiring", "walk-in"), salary mentions and location names are stripped from titles, and legal suffixes (Pvt Ltd, LLP, …) are removed from company names.
- A final cell prints sanity checks (disclosed jobs with zero salary, minimum above maximum salary, empty skill lists).

### 2. Feature building (`train_model.py`)

Two separate feature sets are built:

- **Skills** – every comma-separated entry in `finalSkill` is one token (so "machine learning" stays intact), stored as a binary job × skill matrix with IDF weights. Skills must appear in at least 5 listings.
- **Titles** – a TF-IDF (unigrams + bigrams, up to 20,000 terms) over the job title only.

Vocabularies are saved as JSON and matrices as `.npz`. Nothing is pickled, so the artifacts keep working when scikit-learn is upgraded.

### 3. Recommendation (`recommender.py`)

**Filters**

| Filter | Rule |
|---|---|
| Job title | every typed word must appear in the title |
| Experience | `minimumExperience ≤ your years ≤ maximumExperience` |
| Salary | disclosed-salary jobs only: `maximumSalary ≥ expected salary × 0.85` (±15% tolerance) |

**Match score**

```
match = 0.7 × skill coverage + 0.3 × relative skill similarity
```

- **Skill coverage** – the share of your selected skills that the listing actually lists (0 to 1).
- **Relative skill similarity** – IDF-weighted cosine similarity, scaled so the best result in the current search equals 1. Rare skills count more than common ones, and it breaks ties between listings with the same coverage.

If you also type a job title, a title-similarity term (15% of the final score) is blended in. If you search by title only, ranking uses the title TF-IDF similarity. Listings with a score of 0 are never shown.

### 4. Interface (`app.py`)

A Streamlit front-end that loads the artifacts once (`st.cache_resource`) and renders the form and result tables.

## Run it locally

Requires **Python 3.10+**.

```bash
git lfs install
git clone https://github.com/Moyukh-Mondal/Job_Recommender.git
cd Job_Recommender
pip install -r requirements.txt
streamlit run app.py
```

> **Git LFS:** large files (`.csv`, `.xlsx`, `.npz`) are stored with Git LFS. Downloading the repo as a ZIP from GitHub gives you tiny pointer files instead of the real data (and `train_model.py` will tell you so). Clone with Git LFS installed, or run `git lfs pull` inside the clone.

The app loads pre-built artifacts from `artifacts/`, so no training step is needed to run it.

### Regenerating the cleaned data

1. `pip install rapidfuzz pyahocorasick beautifulsoup4 matplotlib seaborn nltk` (plus Jupyter).
2. Put `indian-job-market-dataset-2025.xlsx` in `data/` (or mount Google Drive when using Colab).
3. Run `Notebooks/Job_Recommender_Cleaning.ipynb` top to bottom. It writes `job_cleaned.csv` to the same folder.

### Regenerating the model artifacts

If `data/job_cleaned.csv` changes, rebuild the artifacts:

```bash
python train_model.py
```

Custom paths are supported: `python train_model.py --data path/to/job_cleaned.csv --out artifacts`. This writes `skills_vocab.json`, `skill_matrix.npz`, `title_vocab.json`, `title_tfidf.npz` and a slim `jobs_model_ready.csv`. The skill dropdown is built from the skill vocabulary automatically.

## Project structure

```
├── app.py                          # Streamlit UI (form, styling, results)
├── recommender.py                  # JobRecommender class (loads artifacts, filters, scores)
├── train_model.py                  # Builds the skill and title artifacts from cleaned data
├── requirements.txt
├── artifacts/                      # Pre-built model artifacts
│   ├── skills_vocab.json
│   ├── skill_matrix.npz
│   ├── title_vocab.json
│   ├── title_tfidf.npz
│   └── jobs_model_ready.csv
├── data/                           # Raw and cleaned datasets (Git LFS)
│   ├── indian-job-market-dataset-2025.xlsx
│   └── job_cleaned.csv
└── Notebooks/
    └── Job_Recommender_Cleaning.ipynb   # Data cleaning & feature extraction
```

## Tech stack

- **Python** – pandas, NumPy
- **scikit-learn** – skill/title vectorization and TF-IDF weighting
- **SciPy** – sparse matrices for the skill and title features
- **Streamlit** – web interface and deployment (with custom CSS/JS for the theme)
- **Aho-Corasick** – fast multi-pattern skill extraction from raw text
- **rapidfuzz** – fuzzy matching for location normalization

## Notes and limitations

- `recommender.py` is kept independent of Streamlit so it can be reused or tested standalone.
- The skill dropdown lists only entries from the `finalSkill` column (those in 5+ listings, up to 4,000, most common first). Skills that are rare in the dataset are not offered.
- The 70/30 weights in the match score are hand-picked and have not been tuned against labelled data.
- Streamlit caches the recommender object. After editing `recommender.py`, restart the app or bump the `version` argument of `load_recommender()` in `app.py`.
- Salaries are shown as stored in the dataset after cleaning. Listings without a usable salary appear only in the "without salary info" table.
- The cursor glow and background animation rely on Streamlit's page structure and may need adjusting after a major Streamlit upgrade.

## Ideas for future work

- Add a location filter using the cleaned `location_clean` column.
- Show which of your selected skills each listing covers.
- Add a small set of test searches to measure ranking quality and tune the score weights.
- Try embedding-based similarity (e.g. sentence transformers) alongside the skill matrix.# 💼 Job Recommender

A content-based job recommendation system built on Indian job listings (roughly 87,000 after cleaning and de-duplication). Pick your skills, optionally search by job title, add your experience and expected salary, and get the closest-matching openings, ranked by how well they fit.

**🚀 Live app:** https://job-recommender-web.streamlit.app/

<!-- Add a screenshot: save it as assets/screenshot.png and uncomment the next line -->
<!-- ![App screenshot](assets/screenshot.png) -->

---

## Features

- **Skill picker** – choose from a searchable dropdown instead of typing a skill list. Start typing to filter and pick as many skills as you like. Options are capitalized (e.g. *Machine Learning*, *SQL*).
- **Job title search** – an optional text box that keeps only jobs whose title contains every word you type.
- **Experience and salary filters** – leave either at `0` to skip it.
- **Two result tables** – jobs *with* salary info and jobs *without*, each ranked by match score.
- **Readable results** – compact columns that fit on screen: salary as a range in LPA (e.g. `₹5.4 – 7.2 LPA`), experience as `5 – 10 yrs`, and the match shown as a percentage bar.
- **Polished UI** – dark theme, glass-style form card, a slowly drifting aurora background and a soft cursor glow. Motion is switched off for users who prefer reduced motion.

## How it works

### 1. Data cleaning (`Notebooks/Job_Recommender_Cleaning.ipynb`)

The raw dataset (`indian-job-market-dataset-2025.xlsx`) is cleaned in a notebook:

- **Duplicates** – ID columns are dropped first, then exact duplicates are removed. After title and company cleaning, postings with the same title, company, location, salary range and experience range are de-duplicated again (the one with the most skills is kept).
- **Salary** – free-text salaries are turned into numeric minimum and maximum values. Monthly salaries (`15,000/month`) are annualised, USD salaries are converted to INR at both ends of the range, and a posting only counts as *disclosed* if its maximum salary is above 0.
- **Skills** – skills come from the structured tags, falling back to skills found in the job description with an **Aho-Corasick automaton**. Both go through the same filter: a skill must appear in at least 5 listings, and generic words, job-title words, education boilerplate and city names are removed.
- **Locations** – location strings are normalized with **fuzzy matching** (rapidfuzz) plus a hand-checked spelling dictionary.
- **Titles and companies** – recruiter noise ("urgent hiring", "walk-in"), salary mentions and location names are stripped from titles, and legal suffixes (Pvt Ltd, LLP, …) are removed from company names.
- A final cell prints sanity checks (disclosed jobs with zero salary, minimum above maximum salary, empty skill lists).

### 2. Feature building (`train_model.py`)

Two separate feature sets are built:

- **Skills** – every comma-separated entry in `finalSkill` is one token (so "machine learning" stays intact), stored as a binary job × skill matrix with IDF weights. Skills must appear in at least 5 listings.
- **Titles** – a TF-IDF (unigrams + bigrams, up to 20,000 terms) over the job title only.

Vocabularies are saved as JSON and matrices as `.npz`. Nothing is pickled, so the artifacts keep working when scikit-learn is upgraded.

### 3. Recommendation (`recommender.py`)

**Filters**

| Filter | Rule |
|---|---|
| Job title | every typed word must appear in the title |
| Experience | `minimumExperience ≤ your years ≤ maximumExperience` |
| Salary | disclosed-salary jobs only: `maximumSalary ≥ expected salary × 0.85` (±15% tolerance) |

**Match score**

```
match = 0.7 × skill coverage + 0.3 × relative skill similarity
```

- **Skill coverage** – the share of your selected skills that the listing actually lists (0 to 1).
- **Relative skill similarity** – IDF-weighted cosine similarity, scaled so the best result in the current search equals 1. Rare skills count more than common ones, and it breaks ties between listings with the same coverage.

If you also type a job title, a title-similarity term (15% of the final score) is blended in. If you search by title only, ranking uses the title TF-IDF similarity. Listings with a score of 0 are never shown.

### 4. Interface (`app.py`)

A Streamlit front-end that loads the artifacts once (`st.cache_resource`) and renders the form and result tables.

## Run it locally

Requires **Python 3.10+**.

```bash
git lfs install
git clone https://github.com/Moyukh-Mondal/Job_Recommender.git
cd Job_Recommender
pip install -r requirements.txt
streamlit run app.py
```

> **Git LFS:** large files (`.csv`, `.xlsx`, `.npz`) are stored with Git LFS. Downloading the repo as a ZIP from GitHub gives you tiny pointer files instead of the real data (and `train_model.py` will tell you so). Clone with Git LFS installed, or run `git lfs pull` inside the clone.

The app loads pre-built artifacts from `artifacts/`, so no training step is needed to run it.

### Regenerating the cleaned data

1. `pip install rapidfuzz pyahocorasick beautifulsoup4 matplotlib seaborn nltk` (plus Jupyter).
2. Put `indian-job-market-dataset-2025.xlsx` in `data/` (or mount Google Drive when using Colab).
3. Run `Notebooks/Job_Recommender_Cleaning.ipynb` top to bottom. It writes `job_cleaned.csv` to the same folder.

### Regenerating the model artifacts

If `data/job_cleaned.csv` changes, rebuild the artifacts:

```bash
python train_model.py
```

Custom paths are supported: `python train_model.py --data path/to/job_cleaned.csv --out artifacts`. This writes `skills_vocab.json`, `skill_matrix.npz`, `title_vocab.json`, `title_tfidf.npz` and a slim `jobs_model_ready.csv`. The skill dropdown is built from the skill vocabulary automatically.

## Project structure

```
├── app.py                          # Streamlit UI (form, styling, results)
├── recommender.py                  # JobRecommender class (loads artifacts, filters, scores)
├── train_model.py                  # Builds the skill and title artifacts from cleaned data
├── requirements.txt
├── artifacts/                      # Pre-built model artifacts
│   ├── skills_vocab.json
│   ├── skill_matrix.npz
│   ├── title_vocab.json
│   ├── title_tfidf.npz
│   └── jobs_model_ready.csv
├── data/                           # Raw and cleaned datasets (Git LFS)
│   ├── indian-job-market-dataset-2025.xlsx
│   └── job_cleaned.csv
└── Notebooks/
    └── Job_Recommender_Cleaning.ipynb   # Data cleaning & feature extraction
```

## Tech stack

- **Python** – pandas, NumPy
- **scikit-learn** – skill/title vectorization and TF-IDF weighting
- **SciPy** – sparse matrices for the skill and title features
- **Streamlit** – web interface and deployment (with custom CSS/JS for the theme)
- **Aho-Corasick** – fast multi-pattern skill extraction from raw text
- **rapidfuzz** – fuzzy matching for location normalization

## Notes and limitations

- `recommender.py` is kept independent of Streamlit so it can be reused or tested standalone.
- The skill dropdown lists only entries from the `finalSkill` column (those in 5+ listings, up to 4,000, most common first). Skills that are rare in the dataset are not offered.
- The 70/30 weights in the match score are hand-picked and have not been tuned against labelled data.
- Streamlit caches the recommender object. After editing `recommender.py`, restart the app or bump the `version` argument of `load_recommender()` in `app.py`.
- Salaries are shown as stored in the dataset after cleaning. Listings without a usable salary appear only in the "without salary info" table.
- The cursor glow and background animation rely on Streamlit's page structure and may need adjusting after a major Streamlit upgrade.

## Ideas for future work

- Add a location filter using the cleaned `location_clean` column.
- Show which of your selected skills each listing covers.
- Add a small set of test searches to measure ranking quality and tune the score weights.
- Try embedding-based similarity (e.g. sentence transformers) alongside the skill matrix.
