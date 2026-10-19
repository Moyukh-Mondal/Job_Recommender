"""
app.py

Streamlit front-end for the job recommender.
Loads pre-built artifacts once (via st.cache_resource) and serves
recommendations through recommender.JobRecommender.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from recommender import JobRecommender

st.set_page_config(page_title="Job Recommender", page_icon="💼", layout="wide")

# ---------------------------------------------------------------- styling
CUSTOM_CSS = """
<style>
/* ---------- Background: dot grid + slowly drifting aurora glows ---------- */
.stApp {
    background-color: #060914;
    background-image:
        radial-gradient(rgba(148,163,184,0.16) 1px, transparent 1.2px),
        radial-gradient(600px 420px at 15% 10%, rgba(99,102,241,0.38), transparent 70%),
        radial-gradient(560px 400px at 85% 8%, rgba(6,182,212,0.30), transparent 70%),
        radial-gradient(620px 460px at 70% 90%, rgba(168,85,247,0.24), transparent 70%),
        radial-gradient(520px 380px at 5% 85%, rgba(16,185,129,0.18), transparent 70%);
    background-size: 26px 26px, 200% 200%, 200% 200%, 200% 200%, 200% 200%;
    background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat;
    background-attachment: fixed;
    /* static starting positions; JS below moves the glows randomly */
    background-position: 0 0, 0% 0%, 100% 0%, 100% 100%, 0% 100%;
}

/* Transparent top bar, hide footer */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Hide any heading anchor links */
[data-testid="stHeaderActionElements"], .stMarkdown a.anchor-link { display: none !important; }

/* Wider centered content */
.block-container { max-width: 1280px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* ---------- Hero ---------- */
.hero { text-align: center; margin: 0.5rem 0 2rem 0; }
.hero-title {
    font-size: 3.1rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15;
}
.hero-text {
    background: linear-gradient(90deg, #a5b4fc 0%, #67e8f9 60%, #6ee7b7 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 1.08rem; margin-top: 0.6rem; }

/* ---------- Form card (glass effect) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 18px !important;
    padding: 0.6rem 0.8rem;
    box-shadow: 0 12px 44px rgba(0,0,0,0.40);
    backdrop-filter: blur(10px);
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: rgba(10,15,30,0.75) !important;
    border-radius: 10px !important;
}

/* Skill chips */
span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
    color: #fff !important;
    border-radius: 999px !important;
    font-weight: 500;
}

/* Primary button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white; border: none; border-radius: 12px;
    padding: 0.7rem 1rem; font-weight: 700; font-size: 1.05rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.45);
    color: white;
}

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }

/* Result tables */
div[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(148,163,184,0.20);
    box-shadow: 0 8px 30px rgba(0,0,0,0.30);
}

/* "Searching for" chips */
.chip {
    display: inline-block; padding: 3px 12px; margin: 2px 4px 2px 0;
    border-radius: 999px; font-size: 0.85rem; font-weight: 500;
    background: rgba(99,102,241,0.18); color: #c7d2fe;
    border: 1px solid rgba(99,102,241,0.4);
}
.searching-label { color: #94a3b8; font-weight: 600; margin-right: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- cursor glow
# Runs inside the main page (not the component iframe) so it survives reruns.
components.html(
    """
    <script>
    const doc = window.parent.document;
    if (!doc.getElementById("cursor-glow-script")) {
        const s = doc.createElement("script");
        s.id = "cursor-glow-script";
        s.textContent = `
        (function () {
            const glow = document.createElement("div");
            glow.id = "cursor-glow";
            Object.assign(glow.style, {
                position: "fixed", top: "0", left: "0",
                width: "150px", height: "150px", margin: "-75px 0 0 -75px",
                borderRadius: "50%", pointerEvents: "none", zIndex: "9999",
                opacity: "0", transition: "opacity .35s ease",
                background: "radial-gradient(circle, rgba(99,102,241,0.24) 0%, rgba(6,182,212,0.12) 40%, transparent 70%)",
                mixBlendMode: "screen"
            });
            document.body.appendChild(glow);

            let tx = 0, ty = 0, x = 0, y = 0;
            document.addEventListener("mousemove", function (e) {
                tx = e.clientX; ty = e.clientY; glow.style.opacity = "1";
            });
            document.documentElement.addEventListener("mouseleave", function () {
                glow.style.opacity = "0";
            });
            (function tick() {
                x += (tx - x) * 0.15; y += (ty - y) * 0.15;
                glow.style.transform = "translate(" + x + "px," + y + "px)";
                requestAnimationFrame(tick);
            })();
        })();
        `;
        doc.head.appendChild(s);
    }

    // Random aura drift: each glow picks its own random target, speed and pause
    if (!doc.getElementById("aura-drift-script")) {
        const a = doc.createElement("script");
        a.id = "aura-drift-script";
        a.textContent = `
        (function () {
            const app = document.querySelector(".stApp");
            if (!app) return;
            if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

            const rnd = (lo, hi) => lo + Math.random() * (hi - lo);
            const SPEED = 0.01;   // same speed for every glow (% of screen per ms; smaller = slower)
            const starts = [[0, 0], [100, 0], [100, 100], [0, 100]];
            const blobs = starts.map(function (p) {
                return { x: p[0], y: p[1], tx: p[0], ty: p[1],
                         wait: rnd(0, 2500), last: performance.now() };
            });

            function pick(b) {
                b.tx = rnd(0, 100);
                b.ty = rnd(0, 100);
            }

            function tick(now) {
                blobs.forEach(function (b) {
                    const dt = Math.min(now - b.last, 50);
                    b.last = now;
                    if (b.wait > 0) {
                        b.wait -= dt;
                        if (b.wait <= 0) pick(b);
                        return;
                    }
                    const dx = b.tx - b.x, dy = b.ty - b.y;
                    const dist = Math.hypot(dx, dy);
                    if (dist < 0.3) {
                        b.wait = rnd(300, 4000);   // random rest before next move
                        return;
                    }
                    // constant speed, easing out only in the last stretch
                    const v = SPEED * Math.min(1, dist / 12 + 0.1);
                    const step = Math.min(v * dt, dist);
                    b.x += (dx / dist) * step;
                    b.y += (dy / dist) * step;
                });
                app.style.backgroundPosition = "0 0," + blobs.map(function (b) {
                    return b.x.toFixed(2) + "% " + b.y.toFixed(2) + "%";
                }).join(",");
                requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
        })();
        `;
        doc.head.appendChild(a);
    }
    </script>
    """,
    height=0,
)

# Words that should be displayed in a specific form
ACRONYMS = {
    "sql": "SQL", "aws": "AWS", "api": "API", "apis": "APIs", "html": "HTML",
    "css": "CSS", "php": "PHP", "etl": "ETL", "ml": "ML", "ai": "AI",
    "nlp": "NLP", "bi": "BI", "ui": "UI", "ux": "UX", "qa": "QA",
    "sap": "SAP", "erp": "ERP", "crm": "CRM", "seo": "SEO", "hr": "HR",
    "ios": "iOS", "gcp": "GCP", "ci": "CI", "cd": "CD", "devops": "DevOps",
    "javascript": "JavaScript", "typescript": "TypeScript", "mysql": "MySQL",
    "nosql": "NoSQL", "mongodb": "MongoDB", "postgresql": "PostgreSQL",
    "powerbi": "PowerBI", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "dsa": "DSA", "oop": "OOP", "cpp": "C++", "net": ".NET",
}


def pretty(skill: str) -> str:
    """'machine learning' -> 'Machine Learning', 'sql' -> 'SQL'."""
    return " ".join(ACRONYMS.get(w, w.capitalize()) for w in skill.split())


def format_salary(lo, hi) -> str:
    """Compact salary range in lakhs per annum, e.g. '₹5.4 – 7.2 LPA'."""
    if not lo and not hi:
        return "Not specified"
    return f"₹{lo / 1e5:.1f} – {hi / 1e5:.1f} LPA"


def prepare(df: pd.DataFrame, with_salary: bool) -> pd.DataFrame:
    """Turn raw results into compact, display-friendly columns that fit
    on screen without horizontal scrolling."""
    out = pd.DataFrame()
    out["Job Title"] = df["title"].astype(str).str.title()
    out["Company"] = df["companyName"].astype(str).str.title()
    if with_salary:
        out["Salary"] = [
            format_salary(lo, hi)
            for lo, hi in zip(df["minimumSalary"].fillna(0), df["maximumSalary"].fillna(0))
        ]
    lo_exp = df["minimumExperience"].fillna(0).astype(int).astype(str)
    hi_exp = df["maximumExperience"].fillna(0).astype(int).astype(str)
    out["Experience"] = lo_exp + " – " + hi_exp + " yrs"
    out["Match"] = (df["match_score"] * 100).round()
    return out


@st.cache_resource
def load_recommender(version: int = 4):
    # bump `version` whenever recommender.py changes to force a fresh load
    return JobRecommender()


recommender = load_recommender()

# ---------------------------------------------------------------- header
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">💼 <span class="hero-text">Job Recommender</span></div>'
    '<p>Pick your skills, optionally search by job title, and get matching listings.</p>'
    '</div>',
    unsafe_allow_html=True,
)

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
RESULT_CONFIG = {
    "Job Title": st.column_config.TextColumn("Job Title", width="large"),
    "Company": st.column_config.TextColumn("Company", width="medium"),
    "Salary": st.column_config.TextColumn("Salary", width="medium"),
    "Experience": st.column_config.TextColumn("Experience", width="small"),
    "Match": st.column_config.ProgressColumn(
        "Match", min_value=0, max_value=100, format="%d%%", width="small"
    ),
}

st.write("")

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
            chips = "".join(f'<span class="chip">{pretty(s)}</span>' for s in selected_skills)
            st.markdown(
                f'<span class="searching-label">Searching for:</span>{chips}',
                unsafe_allow_html=True,
            )
            st.write("")

        tab1, tab2 = st.tabs(["💰 With salary info", "❔ Without salary info"])

        with tab1:
            if disclosed_df.empty:
                st.info("No matches found in this category.")
            else:
                st.dataframe(
                    prepare(disclosed_df, with_salary=True),
                    column_config=RESULT_CONFIG, hide_index=True, width="stretch",
                )
        with tab2:
            if undisclosed_df.empty:
                st.info("No matches found in this category.")
            else:
                st.dataframe(
                    prepare(undisclosed_df, with_salary=False),
                    column_config=RESULT_CONFIG, hide_index=True, width="stretch",
                )
else:
    st.info("Your results will appear here after you search.")
