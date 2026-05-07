import csv
import hashlib
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hybrid_cnn_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
CONFIG_PATH = BASE_DIR / "config.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
USERS_PATH = BASE_DIR / "users.json"
REVIEWS_PATH = BASE_DIR / "employee_reviews.csv"

DEPARTMENTS = [
    "Human Resources",
    "Finance",
    "Information Technology",
    "Sales",
    "Marketing",
    "Operations",
    "Customer Service",
    "Research and Development",
    "Administration",
]

REVIEW_COLUMNS = [
    "timestamp",
    "username",
    "employee_name",
    "department",
    "review_text",
    "overall_rating",
    "work_life_balance",
    "vader_negative",
    "vader_neutral",
    "vader_positive",
    "vader_compound",
    "sentiment_label",
    "stay_probability",
    "leave_probability",
    "risk_level",
]

REASON_KEYWORDS = {
    "management issues": [
        "management",
        "manager",
        "supervisor",
        "leader",
        "leadership",
        "boss",
        "micromanage",
    ],
    "heavy workload or stress": [
        "workload",
        "stress",
        "stressful",
        "overload",
        "pressure",
        "burnout",
        "burned out",
        "too much work",
        "tired",
        "exhausted",
    ],
    "low salary or poor benefits": [
        "salary",
        "pay",
        "underpaid",
        "compensation",
        "benefit",
        "benefits",
        "bonus",
        "increment",
    ],
    "limited career growth": [
        "career",
        "growth",
        "promotion",
        "promote",
        "development",
        "training",
        "opportunity",
        "opportunities",
    ],
    "poor work-life balance": [
        "work life",
        "work-life",
        "balance",
        "overtime",
        "long hour",
        "long hours",
        "weekend",
        "late night",
    ],
    "negative workplace culture": [
        "culture",
        "toxic",
        "environment",
        "colleague",
        "team",
        "communication",
        "support",
        "unfair",
    ],
    "intention to leave": [
        "leave",
        "leaving",
        "resign",
        "resignation",
        "quit",
        "quitting",
        "looking for another job",
        "new job",
    ],
}


def clean_text(value):
    value = "" if value is None else str(value)
    value = re.sub(r"<.*?>", " ", value)
    value = re.sub(r"http\S+|www\S+", " ", value)
    value = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " ", value)
    value = re.sub(r"[^A-Za-z0-9\s.,!?'-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def password_hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def default_users():
    return {
        "hr": {
            "password_hash": password_hash("admin123"),
            "role": "HR",
            "name": "HR Admin",
            "department": "Human Resources",
        },
        "employee": {
            "password_hash": password_hash("employee123"),
            "role": "Employee",
            "name": "Demo Employee",
            "department": "Information Technology",
        },
    }


def supabase_config():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def use_supabase():
    return supabase_config() is not None


def supabase_headers():
    config = supabase_config()
    return {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def supabase_table_url(table):
    return f"{supabase_config()['url']}/rest/v1/{table}"


def supabase_get(table, params=None):
    response = requests.get(
        supabase_table_url(table),
        headers=supabase_headers(),
        params=params or {"select": "*"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def supabase_upsert(table, rows, conflict_column):
    response = requests.post(
        supabase_table_url(table),
        headers={**supabase_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
        params={"on_conflict": conflict_column},
        json=rows,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def supabase_insert(table, row):
    response = requests.post(
        supabase_table_url(table),
        headers=supabase_headers(),
        json=row,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def ensure_storage():
    if use_supabase():
        users = [
            {"username": username, **data}
            for username, data in default_users().items()
        ]
        supabase_upsert("users", users, "username")
        return

    if not USERS_PATH.exists():
        USERS_PATH.write_text(json.dumps(default_users(), indent=2), encoding="utf-8")

    if not REVIEWS_PATH.exists():
        with REVIEWS_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()


def load_users():
    ensure_storage()
    if use_supabase():
        rows = supabase_get("users", {"select": "username,password_hash,role,name,department"})
        return {
            row["username"]: {
                "password_hash": row["password_hash"],
                "role": row["role"],
                "name": row["name"],
                "department": row["department"],
            }
            for row in rows
        }
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))


def save_users(users):
    if use_supabase():
        rows = [{"username": username, **data} for username, data in users.items()]
        supabase_upsert("users", rows, "username")
        return
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def load_reviews():
    ensure_storage()
    if use_supabase():
        rows = supabase_get(
            "employee_reviews",
            {"select": ",".join(REVIEW_COLUMNS), "order": "timestamp.desc"},
        )
        return pd.DataFrame(rows, columns=REVIEW_COLUMNS)

    if REVIEWS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return pd.read_csv(REVIEWS_PATH)


def append_review(row):
    ensure_storage()
    if use_supabase():
        supabase_insert("employee_reviews", row)
        return

    with REVIEWS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
        writer.writerow(row)


def logout():
    st.session_state.authenticated = False
    st.session_state.pop("user", None)


@st.cache_resource
def load_artifacts():
    missing = [
        str(path.name)
        for path in [MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH, SCALER_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    with CONFIG_PATH.open("rb") as file:
        config = pickle.load(file)

    return {
        "model": load_model(MODEL_PATH),
        "tokenizer": tokenizer,
        "maxlen": int(config.get("maxlen", 100)),
        "scaler": joblib.load(SCALER_PATH),
        "vader": SentimentIntensityAnalyzer(),
    }


def sentiment_label(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def risk_label(leave_probability):
    if leave_probability >= 0.55:
        return "High"
    if leave_probability >= 0.35:
        return "Medium"
    return "Low"


def calibrated_leave_probability(model_leave_probability, vader_compound, overall_rating, work_life_balance):
    sentiment_risk = 1 - ((vader_compound + 1) / 2)
    rating_risk = ((5 - overall_rating) + (5 - work_life_balance)) / 8
    leave_probability = (
        0.45 * model_leave_probability
        + 0.35 * sentiment_risk
        + 0.20 * rating_risk
    )
    return float(np.clip(leave_probability, 0, 1))


def high_risk_reason(row):
    review_text = clean_text(row.get("review_text", ""))
    reasons = [
        reason
        for reason, keywords in REASON_KEYWORDS.items()
        if any(keyword in review_text for keyword in keywords)
    ]

    if reasons:
        return "Comment indicates " + ", ".join(reasons) + "."
    if row.get("sentiment_label") == "Negative":
        return "Comment has negative sentiment, but no specific issue keyword was detected."
    return "Comment pattern is similar to reviews with high leave potential."


def predict_review(review_text, overall_rating, work_life_balance):
    artifacts = load_artifacts()
    cleaned = clean_text(review_text)
    scores = artifacts["vader"].polarity_scores(cleaned)
    sequence = artifacts["tokenizer"].texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=artifacts["maxlen"],
        padding="post",
        truncating="post",
    )
    vader_input = np.array([[scores["compound"]]], dtype="float32")
    extra_features = artifacts["scaler"].transform(
        np.array([[overall_rating, work_life_balance]], dtype="float32")
    )
    stay_probability = float(
        artifacts["model"].predict([padded, vader_input, extra_features], verbose=0)[0][0]
    )
    model_leave_probability = 1.0 - stay_probability
    leave_probability = calibrated_leave_probability(
        model_leave_probability,
        scores["compound"],
        overall_rating,
        work_life_balance,
    )

    return {
        "vader_negative": float(scores["neg"]),
        "vader_neutral": float(scores["neu"]),
        "vader_positive": float(scores["pos"]),
        "vader_compound": float(scores["compound"]),
        "sentiment_label": sentiment_label(scores["compound"]),
        "stay_probability": 1.0 - leave_probability,
        "leave_probability": leave_probability,
        "risk_level": risk_label(leave_probability),
    }


def percent_label(value):
    return f"{float(value) * 100:.1f}%"


def render_page_header(title, subtitle=None, eyebrow=None):
    eyebrow_html = f"<div class='eyebrow'>{eyebrow}</div>" if eyebrow else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <section class="page-hero">
            {eyebrow_html}
            <h1>{title}</h1>
            {subtitle_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_account_bar(user):
    account_col, logout_col = st.columns([5, 1])
    account_col.markdown(
        f"""
        <div class="account-strip">
            <span class="account-name">{user['name']}</span>
            <span>{user['role']} | {user.get('department', 'No department')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    logout_col.button("Logout", key="logout_button", use_container_width=True, on_click=logout)


def login_view():
    render_page_header(
        "Employee Retention Prediction System",
        "A focused workspace for collecting employee feedback, measuring sentiment, and reviewing retention risk.",
        "Workforce Intelligence",
    )

    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", use_container_width=True):
            users = load_users()
            user = users.get(username.strip().lower())
            if user and user["password_hash"] == password_hash(password):
                st.session_state.user = {"username": username.strip().lower(), **user}
                st.session_state.authenticated = True
                st.session_state.pop("logout_handled", None)
                st.query_params.clear()
                st.rerun()
            st.error("Invalid username or password.")

    with tab_register:
        name = st.text_input("Full name")
        new_username = st.text_input("New username")
        new_password = st.text_input("New password", type="password")
        role = st.selectbox("Account type", ["Employee", "HR"])
        department = st.selectbox("Department", DEPARTMENTS)

        if st.button("Create account", use_container_width=True):
            users = load_users()
            username = new_username.strip().lower()
            if not name.strip() or not username or not new_password:
                st.error("Please complete all fields.")
            elif username in users:
                st.error("This username already exists.")
            else:
                users[username] = {
                    "password_hash": password_hash(new_password),
                    "role": role,
                    "name": name.strip(),
                    "department": department,
                }
                save_users(users)
                st.success("Account created. You can login now.")


def require_login(required_role=None):
    user = st.session_state.get("user")
    is_authenticated = bool(st.session_state.get("authenticated")) and user is not None
    if not is_authenticated:
        return None
    if required_role and user.get("role") != required_role:
        return None
    return user


def employee_view(user):
    if require_login("Employee") is None:
        st.stop()

    render_page_header(
        "Employee Review",
        "Share workplace feedback so HR can identify retention risk and support teams earlier.",
        "Feedback Submission",
    )

    with st.form("review_form"):
        employee_name = st.text_input("Employee name", value=user.get("name", ""))
        department = st.selectbox(
            "Department",
            DEPARTMENTS,
            index=DEPARTMENTS.index(user["department"]) if user.get("department") in DEPARTMENTS else 0,
        )
        review = st.text_area(
            "Review",
            height=180,
            placeholder="Write your honest review about workload, management, salary, culture, benefits, or career growth.",
        )
        col_a, col_b = st.columns(2)
        overall_rating = col_a.slider("Overall rating", 1, 5, 3)
        work_life_balance = col_b.slider("Work-life balance", 1, 5, 3)
        submitted = st.form_submit_button("Submit Review", type="primary", use_container_width=True)

    if submitted:
        if len(clean_text(review).split()) < 3:
            st.error("Please enter a longer review before submitting.")
            return

        prediction = predict_review(review, overall_rating, work_life_balance)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": user["username"],
            "employee_name": employee_name.strip() or user.get("name", ""),
            "department": department,
            "review_text": clean_text(review),
            "overall_rating": overall_rating,
            "work_life_balance": work_life_balance,
            **prediction,
        }
        append_review(row)

        st.success("Review submitted successfully.")
        col_1, col_2, col_3 = st.columns(3)
        col_1.metric("Sentiment", prediction["sentiment_label"])
        col_2.metric("Potential to Leave", percent_label(prediction["leave_probability"]))
        col_3.metric("Risk Level", prediction["risk_level"])

    reviews = load_reviews()
    own_reviews = reviews[reviews["username"] == user["username"]] if not reviews.empty else reviews
    if not own_reviews.empty:
        st.subheader("My Previous Reviews")
        own_reviews_table = (
            own_reviews[
                [
                    "timestamp",
                    "department",
                    "review_text",
                    "sentiment_label",
                    "leave_probability",
                    "risk_level",
                ]
            ]
            .sort_values("timestamp", ascending=False)
            .rename(
                columns={
                    "timestamp": "Submitted",
                    "department": "Department",
                    "review_text": "Comment",
                    "sentiment_label": "Sentiment",
                    "leave_probability": "Leave Probability",
                    "risk_level": "Risk Level",
                }
            )
        )
        own_reviews_table["Leave Probability"] = pd.to_numeric(
            own_reviews_table["Leave Probability"],
            errors="coerce",
        ).fillna(0) * 100
        st.dataframe(
            own_reviews_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Leave Probability": st.column_config.ProgressColumn(
                    "Leave Probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )


def empty_dashboard():
    st.info("No employee reviews have been submitted yet. Ask employees to submit reviews first.")


def hr_view():
    if require_login("HR") is None:
        st.stop()

    render_page_header(
        "HR Dashboard",
        "Monitor employee sentiment, department-level retention signals, and the people who need attention first.",
        "Decision Dashboard",
    )

    reviews = load_reviews()
    if reviews.empty:
        empty_dashboard()
        return

    reviews["leave_probability"] = pd.to_numeric(reviews["leave_probability"], errors="coerce").fillna(0)
    reviews["stay_probability"] = pd.to_numeric(reviews["stay_probability"], errors="coerce").fillna(0)

    departments = ["All Departments"] + sorted(reviews["department"].dropna().unique().tolist())
    selected_department = st.selectbox("Department filter", departments)
    filtered = reviews if selected_department == "All Departments" else reviews[reviews["department"] == selected_department]

    if filtered.empty:
        st.warning("No reviews found for this department.")
        return

    high_risk = filtered[filtered["risk_level"] == "High"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Reviews", len(filtered))
    col_b.metric("Average Leave Potential", percent_label(filtered["leave_probability"].mean()))
    col_c.metric("High Risk Employees", len(high_risk))
    col_d.metric("Negative Sentiment", int((filtered["sentiment_label"] == "Negative").sum()))

    chart_a, chart_b = st.columns(2)
    sentiment_counts = filtered["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    sentiment_chart = (
        alt.Chart(sentiment_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#2f9e44", "#64748b", "#d9480f"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Sentiment", "Count"],
        )
        .properties(title="Sentiment Analysis", height=320)
        .configure_title(anchor="start", color="#102a43", fontSize=16)
    )
    chart_a.altair_chart(sentiment_chart, use_container_width=True)

    risk_counts = filtered["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_chart = (
        alt.Chart(risk_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Risk Level:N",
                scale=alt.Scale(
                    domain=["Low", "Medium", "High"],
                    range=["#2f9e44", "#f59f00", "#d9480f"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Risk Level", "Count"],
        )
        .properties(title="Employee Retention Prediction", height=320)
        .configure_title(anchor="start", color="#102a43", fontSize=16)
    )
    chart_b.altair_chart(risk_chart, use_container_width=True)

    dept_summary = (
        reviews.groupby("department", as_index=False)
        .agg(
            employees=("username", "count"),
            average_leave_probability=("leave_probability", "mean"),
            high_risk_count=("risk_level", lambda value: int((value == "High").sum())),
        )
        .sort_values("average_leave_probability", ascending=False)
    )
    dept_summary["average_leave_percent"] = dept_summary["average_leave_probability"] * 100

    st.subheader("Department Leave Potential")
    bar_chart = (
        alt.Chart(dept_summary)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("department:N", sort="-y", title="Department", axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("average_leave_percent:Q", title="Average Potential to Leave (%)"),
            color=alt.Color(
                "high_risk_count:Q",
                title="High Risk Count",
                scale=alt.Scale(range=["#9ecae1", "#d9480f"]),
            ),
            tooltip=[
                alt.Tooltip("department:N", title="Department"),
                alt.Tooltip("average_leave_percent:Q", title="Average Leave %", format=".1f"),
                alt.Tooltip("high_risk_count:Q", title="High Risk Count"),
                alt.Tooltip("employees:Q", title="Reviews"),
            ],
        )
        .properties(height=360)
        .configure_view(stroke=None)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    highest_department = dept_summary.iloc[0]
    st.warning(
        f"Highest potential to leave: {highest_department['department']} "
        f"({highest_department['average_leave_percent']:.1f}% average leave probability)."
    )

    st.subheader("Employees With High Potential to Leave")
    high_risk_table = filtered[filtered["risk_level"] == "High"].sort_values(
        "leave_probability",
        ascending=False,
    )
    if high_risk_table.empty:
        st.info("No high-risk employees found for the selected department.")
    else:
        high_risk_table = (
            high_risk_table[
                [
                    "timestamp",
                    "employee_name",
                    "department",
                    "review_text",
                    "sentiment_label",
                    "overall_rating",
                    "work_life_balance",
                    "leave_probability",
                    "risk_level",
                ]
            ].rename(
                columns={
                    "timestamp": "Submitted",
                    "employee_name": "Employee",
                    "department": "Department",
                    "review_text": "Comment",
                    "sentiment_label": "Sentiment",
                    "overall_rating": "Overall Rating",
                    "work_life_balance": "Work-Life Balance",
                    "leave_probability": "Leave Probability",
                    "risk_level": "Risk Level",
                }
            )
        )
        high_risk_table["Leave Probability"] = pd.to_numeric(
            high_risk_table["Leave Probability"],
            errors="coerce",
        ).fillna(0) * 100
        st.dataframe(
            high_risk_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Leave Probability": st.column_config.ProgressColumn(
                    "Leave Probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )


def main():
    st.set_page_config(
        page_title="Employee Retention Prediction",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
            --primary: #0f6b6e;
            --primary-dark: #0b4f52;
            --surface: #ffffff;
            --surface-soft: #f6f8fb;
            --border: #d9e2ec;
            --text: #102a43;
            --muted: #627d98;
            --accent: #f59f00;
            --danger: #d9480f;
        }

        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }

        [data-testid="stAppViewContainer"] > .main {
            margin-left: 0;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(15, 107, 110, 0.08), rgba(246, 248, 251, 0) 280px),
                var(--surface-soft);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .page-hero {
            padding: 1.4rem 0 1rem;
            border-bottom: 1px solid rgba(16, 42, 67, 0.12);
            margin-bottom: 1.3rem;
        }

        .page-hero h1 {
            color: var(--text);
            font-size: 3.25rem;
            line-height: 1.05;
            letter-spacing: 0;
            margin: 0;
        }

        .page-hero p {
            color: var(--muted);
            font-size: 1.05rem;
            max-width: 760px;
            margin: 0.75rem 0 0;
        }

        .eyebrow {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .account-strip {
            align-items: center;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--muted);
            display: flex;
            gap: 0.7rem;
            min-height: 44px;
            padding: 0.55rem 0.8rem;
        }

        .account-name {
            color: var(--text);
            font-weight: 700;
        }

        div[data-testid="stTabs"] button {
            color: var(--muted);
            font-weight: 700;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--primary);
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.07);
            padding: 1rem 1.1rem;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.75rem;
        }

        .stButton button,
        .stFormSubmitButton button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.75rem;
        }

        .stButton button[kind="primary"],
        .stFormSubmitButton button[kind="primary"] {
            background: var(--primary);
            border-color: var(--primary);
        }

        .stButton button[kind="primary"]:hover,
        .stFormSubmitButton button[kind="primary"]:hover {
            background: var(--primary-dark);
            border-color: var(--primary-dark);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 8px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        h2, h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }

            .page-hero {
                padding-top: 0.8rem;
            }

            .page-hero h1 {
                font-size: 2rem;
            }

            .account-strip {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.15rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    ensure_storage()

    page = st.empty()
    sidebar = st.sidebar.empty()

    user = require_login()
    if user is None:
        st.session_state.authenticated = False
        sidebar.empty()
        with page.container():
            login_view()
        st.stop()

    sidebar.empty()

    with page.container():
        render_account_bar(user)

        if user["role"] == "Employee":
            try:
                load_artifacts()
            except Exception as exc:
                st.error(f"Unable to load model artifacts: {exc}")
                st.stop()

        if user["role"] == "HR":
            hr_view()
        else:
            employee_view(user)


if __name__ == "__main__":
    main()
