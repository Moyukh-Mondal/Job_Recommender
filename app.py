"""
app.py

Streamlit front-end for the job recommender.
Loads pre-built artifacts once (via st.cache_resource) and serves
recommendations through recommender.JobRecommender.
"""

import streamlit as st
from recommender import JobRecommender

st.set_page_config(page_title="Job Recommender", page_icon="💼", layout="wide")

# Words that should be fully upper-case when displayed
ACRONYMS = {
    "sql": "SQL", "aws": "AWS", "api": "API", "apis": "APIs", "html": "HTML",
    "css": "CSS", "php": "PHP", "etl": "ETL", "ml": "ML", "ai": "AI",
    "nlp": "NLP", "bi": "BI", "ui": "UI", "ux": "UX", "qa": "QA",
    "sap": "SAP", "erp": "ERP", "crm": "CRM", "seo": "SEO", "hr": "HR",
    "ios": "iOS", "gcp": "GCP", "ci": "CI", "cd": "CD", "devops": "DevOps",
    "javascript": "JavaScript", "typescript": "TypeScript", "mysql": "MySQL",
    "nosql": "NoSQL", "mongodb": "MongoDB", "postgresql": "PostgreSQL",
    "powerbi": "PowerBI", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "aws": "AWS", "dsa": "DSA", "oop": "OOP", "cpp": "C++", "net": ".NET",
}


def pretty(skill: str) -> str:
    """'machine learning' -> 'Machine Learning', 'sql' -> 'SQL'."""
    return " ".join(ACRONYMS.get(w, w.capitalize()) for w in skill.split())


@st.cache_resource
def load_recommender(version: int = 3):
    # bump `version` whenever recommender.py changes to force a reload
    return JobRecommender()


recommender = load_recommender()

# ---------------------------------------------------------------- header
st.title("💼 Job Recommender")
st.caption("Pick your skills, optionally search by job title, and get matching listings.")

# ---------------------------------------------------------------- search form
with st.container(border=True):
    with st.form("search_form", border=False):
        selected_skills = st.multiselect(
            "🛠️ Skills",
            options=recommender.skill_options,
            format_func=pretty,
            placeholder="Click or type to search skills (e.g. Python, SQL, Machine Learning)",
            help="Start typing to filter the list, then pick as many skills as you like.",
        )

        title_input = st.text_input(
            "🔎 Job title search (optional)",
            placeholder="e.g. data analyst, backend developer",
            help="Only jobs whose title contains all of these words are shown.",
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            experience_input = st.number_input(
                "🎓 Years of experience", min_value=0, max_value=40, value=0, step=1,
                help="Leave at 0 to skip the experience filter.",
            )
        with col2:
            salary_input = st.number_input(
                "💰 Expected annual salary (₹)", min_value=0, value=0, step=50000,
                help="Leave at 0 to skip the salary filter.",
            )
        with col3:
            top_n = st.slider("📋 Results per category", 5, 30, 10)

        submitted = st.form_submit_button(
            "Find jobs", type="primary", use_container_width=True
        )

# ---------------------------------------------------------------- results
DISCLOSED_CONFIG = {
    "title": st.column_config.TextColumn("Job Title", width="large"),
    "companyName": st.column_config.TextColumn("Company"),
    "minimumSalary": st.column_config.NumberColumn("Min Salary", format="₹%d"),
    "maximumSalary": st.column_config.NumberColumn("Max Salary", format="₹%d"),
    "minimumExperience": st.column_config.NumberColumn("Min Exp (yrs)"),
    "maximumExperience": st.column_config.NumberColumn("Max Exp (yrs)"),
    "match_score": st.column_config.ProgressColumn(
        "Match", min_value=0.0, max_value=1.0, format="%.2f"
    ),
}

if submitted:
    skills_query = " ".join(selected_skills)
    title_query = title_input.strip()

    if not skills_query and not title_query:
        st.warning("Please select at least one skill or enter a job title.")
    else:
        with st.spinner("Finding matches..."):
            disclosed_df, undisclosed_df = recommender.recommend(
                user_skills=skills_query,
                user_experience=experience_input if experience_input > 0 else None,
                user_expected_salary=salary_input if salary_input > 0 else None,
                title_query=title_query,
                selected_skills=selected_skills,
                top_n=top_n,
            )

        if selected_skills:
            st.markdown(
                "**Searching for:** " + " ".join(f"`{pretty(s)}`" for s in selected_skills)
            )

        tab1, tab2 = st.tabs(["💰 With salary info", "❔ Without salary info"])

        with tab1:
            if disclosed_df.empty:
                st.info("No matches found in this category.")
            else:
                st.dataframe(
                    disclosed_df,
                    column_config=DISCLOSED_CONFIG,
                    hide_index=True,
                    width="stretch",
                )
        with tab2:
            if undisclosed_df.empty:
                st.info("No matches found in this category.")
            else:
                st.dataframe(
                    undisclosed_df,
                    column_config=DISCLOSED_CONFIG,
                    hide_index=True,
                    width="stretch",
                )
else:
    st.info("Your results will appear here after you search.")
