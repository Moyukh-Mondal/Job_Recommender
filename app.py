"""
app.py

Streamlit front-end for the job recommender.
Loads pre-built artifacts once (via st.cache_resource) and serves
recommendations through recommender.JobRecommender.
"""

import streamlit as st
from recommender import JobRecommender

st.set_page_config(page_title="Job Recommender", page_icon="💼", layout="wide")


@st.cache_resource
def load_recommender():
    return JobRecommender()


recommender = load_recommender()

st.title("💼 Job Recommender")
st.write(
    "Enter your skills, experience, and expected salary to get matching job listings."
)

with st.form("search_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        skills_input = st.text_input(
            "Skills",
            placeholder="e.g. python sql machine learning data analysis",
        )
    with col2:
        experience_input = st.number_input(
            "Years of experience", min_value=0, max_value=40, value=0, step=1
        )
    with col3:
        salary_input = st.number_input(
            "Expected annual salary (INR)", min_value=0, value=0, step=50000
        )

    top_n = st.slider("Number of results per category", min_value=5, max_value=30, value=10)

    submitted = st.form_submit_button("Find jobs")

if submitted:
    if not skills_input.strip():
        st.warning("Please enter at least one skill to search.")
    else:
        with st.spinner("Finding matches..."):
            disclosed_df, undisclosed_df = recommender.recommend(
                user_skills=skills_input,
                user_experience=experience_input if experience_input > 0 else None,
                user_expected_salary=salary_input if salary_input > 0 else None,
                top_n=top_n,
            )

        st.subheader(f"Jobs with salary info ({len(disclosed_df)})")
        if disclosed_df.empty:
            st.info("No matches found in this category.")
        else:
            st.dataframe(disclosed_df, width='stretch')

        st.subheader(f"Jobs without salary info ({len(undisclosed_df)})")
        if undisclosed_df.empty:
            st.info("No matches found in this category.")
        else:
            st.dataframe(undisclosed_df, width='stretch')
else:
    st.caption("Results will appear here after you search.")
