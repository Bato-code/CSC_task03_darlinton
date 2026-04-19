import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector – CSC 309",
    page_icon="📧",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* ── Sky Blue Background ── */
.stApp {
    background: linear-gradient(135deg, #87CEEB 0%, #b3e5fc 50%, #e1f5fe 100%) !important;
    min-height: 100vh;
}

/* Main content card */
.main > .block-container {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.7) !important;
    box-shadow: 0 8px 40px rgba(2, 119, 189, 0.18) !important;
    padding: 2rem 2.5rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* ── Global text visibility ── */
/* Force dark navy on all text so it's readable on every background */
p, span, div, li {
    color: #01579b;
}
label,
.stTextArea label,
.stRadio label,
.stMarkdown p,
.stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stWidgetLabel"] p,
.stRadio > div > label > div > p,
.stRadio [role="radiogroup"] label p {
    color: #01579b !important;
    font-weight: 500 !important;
}
/* Alert text stays dark */
.stAlert p { color: #1a1a2e !important; }

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #01579b;
    margin-bottom: 0.3rem;
    text-shadow: 0 2px 8px rgba(1, 87, 155, 0.15);
    line-height: 1.2;
}
.main-header p {
    color: #0277bd;
    font-size: 0.95rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(2, 119, 189, 0.12);
    color: #01579b;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Textarea ── */
.stTextArea textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    border-radius: 12px !important;
    border: 1.5px solid rgba(2, 119, 189, 0.3) !important;
    padding: 14px !important;
    background: rgba(255, 255, 255, 0.98) !important;
    color: #01579b !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    width: 100% !important;
    box-sizing: border-box !important;
}
.stTextArea textarea:focus {
    border-color: #0288d1 !important;
    box-shadow: 0 0 0 3px rgba(2, 136, 209, 0.2) !important;
    background: #fff !important;
}
.stTextArea textarea::placeholder {
    color: #90caf9 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0288d1, #01579b) !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 15px rgba(2, 119, 189, 0.35) !important;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #039be5, #0277bd) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(2, 119, 189, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Divider ── */
hr {
    border-color: rgba(2, 119, 189, 0.2) !important;
}

/* ── Result cards ── */
.result-spam {
    background: rgba(255, 235, 235, 0.98);
    border: 1.5px solid #fca5a5;
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    backdrop-filter: blur(8px);
}
.result-ham {
    background: rgba(235, 253, 240, 0.98);
    border: 1.5px solid #86efac;
    border-left: 5px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    backdrop-filter: blur(8px);
}
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: #1a1a2e !important;
}
.result-sub {
    font-size: 0.9rem;
    color: #0277bd !important;
    margin-bottom: 1rem;
    font-weight: 600;
}
.confidence-bar-wrap {
    background: rgba(2, 119, 189, 0.15);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 0.4rem;
}
.confidence-bar-spam {
    background: linear-gradient(90deg, #f87171, #ef4444);
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.confidence-bar-ham {
    background: linear-gradient(90deg, #4ade80, #22c55e);
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.conf-label {
    font-size: 0.82rem;
    color: #01579b !important;
    font-weight: 600;
}

/* ── Example chips ── */
.example-chip {
    display: inline-block;
    background: rgba(2, 119, 189, 0.1);
    border: 1px solid rgba(2, 119, 189, 0.25);
    border-radius: 999px;
    font-size: 0.8rem;
    padding: 5px 13px;
    margin: 3px 3px 3px 0;
    cursor: pointer;
    color: #01579b;
    transition: background 0.15s;
}
.example-chip:hover { background: rgba(2, 119, 189, 0.2); }

/* ── Concept pills ── */
.concept-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 1rem 0;
}
.concept-pill {
    background: rgba(2, 119, 189, 0.12);
    color: #01579b;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 5px 12px;
}

/* ── Sidebar ── */
.sidebar-section {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.sidebar-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #0288d1;
    margin-bottom: 0.6rem;
}

/* ════════════════════════════════════════
   MOBILE RESPONSIVENESS
   ════════════════════════════════════════ */

/* Tablet (≤768px) */
@media (max-width: 768px) {
    .main > .block-container {
        padding: 1.2rem 1rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        border-radius: 14px !important;
    }
    .main-header h1  { font-size: 1.7rem; }
    .main-header p   { font-size: 0.85rem; }
    .result-label    { font-size: 1.2rem; }
    .result-spam,
    .result-ham      { padding: 1.1rem; }
    .stButton > button {
        font-size: 0.95rem !important;
        padding: 0.65rem !important;
    }
    .stTextArea textarea { font-size: 0.82rem !important; }
}

/* Mobile (≤480px) */
@media (max-width: 480px) {
    .main > .block-container {
        padding: 0.9rem 0.7rem !important;
        margin-top: 0.25rem !important;
        border-radius: 10px !important;
    }
    .main-header          { padding: 1rem 0 0.8rem; }
    .main-header h1       { font-size: 1.25rem; letter-spacing: -0.01em; }
    .main-header p        { font-size: 0.78rem; }
    .result-label         { font-size: 1rem; }
    .result-spam,
    .result-ham           { padding: 0.9rem; }
    .stTextArea textarea  { font-size: 0.78rem !important; }
    .stButton > button    { font-size: 0.88rem !important; }
    .conf-label           { font-size: 0.75rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Data & Model Logic (Cached) ────────────────────────────────────────────────
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

@st.cache_data(show_spinner=False)
def load_and_prep_data():
    try:
        df = pd.read_csv(DATASET_URL, sep='\t', header=None, names=['label', 'message'])
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

@st.cache_resource(show_spinner=False)
def train_models():
    df = load_and_prep_data()
    if df is None:
        return None, None
    
    X = df['message']
    y = df['label_num']
    
    # Simple split for demonstration (though we train on all for better app performance)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Naive Bayes Pipeline
    nb_pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    nb_pipe.fit(X, y)
    
    # Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', LogisticRegression())
    ])
    lr_pipe.fit(X, y)
    
    return nb_pipe, lr_pipe


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📧 Spam Email Detector</h1>
    <p>Analyze any message below to detect if it is spam using AI</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Text input ────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Input your message",
    value="",
    height=180,
    placeholder="Paste or type a message here..."
)

# ── Algorithm Toggle (Below Text Area) ────────────────────────────────────────
st.markdown("Select Detection Algorithm:")
algo_choice = st.radio(
    "Algorithm Selection:",
    ["Naive Bayes", "Logistic Regression"],
    horizontal=True,
    label_visibility="collapsed"
)

# ── Analyse button ────────────────────────────────────────────────────────────
analyse = st.button("Check Spam", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if analyse:
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        with st.spinner(f"Analysing with {algo_choice}..."):
            nb_model, lr_model = train_models()
            
            if nb_model is None:
                st.error("Could not load data or train models.")
            else:
                model = nb_model if algo_choice == "Naive Bayes" else lr_model
                
                # Predict
                prediction = model.predict([user_input])[0]
                probabilities = model.predict_proba([user_input])[0] # [P(0), P(1)]
                
                is_spam = prediction == 1
                score = probabilities[1] if is_spam else probabilities[0]
                
                confidence = round(score * 100, 1)
                opposite = round((1 - score) * 100, 1)

                if is_spam:
                    st.markdown(f"""
        <div class="result-spam">
            <div class="result-label">🔴 SPAM DETECTED</div>
            <div class="result-sub">Algorithm: <strong>{algo_choice}</strong></div>
            <div class="conf-label">Spam confidence: <strong>{confidence}%</strong></div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-spam" style="width:{confidence}%"></div>
            </div>
            <div class="conf-label" style="margin-top:8px;">Safe probability: {opposite}%</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-ham" style="width:{opposite}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
        <div class="result-ham">
            <div class="result-label">✅ NOT SPAM — Looks Safe</div>
            <div class="result-sub">Algorithm: <strong>{algo_choice}</strong></div>
            <div class="conf-label">Safe confidence: <strong>{confidence}%</strong></div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-ham" style="width:{confidence}%"></div>
            </div>
            <div class="conf-label" style="margin-top:8px;">Spam probability: {opposite}%</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-spam" style="width:{opposite}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()

st.markdown("""
<div style="
    text-align: center;
    background: rgba(1, 87, 155, 0.07);
    border: 1px solid rgba(2, 119, 189, 0.2);
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    margin-top: 0.4rem;
">
    <span style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:#0288d1;">🎓 Owner &nbsp;|&nbsp;</span>
    <span style="font-size:0.72rem; color:#0277bd; font-weight:600;">Chukwudi Darlington</span>
    <span style="font-size:0.65rem; color:#90caf9;"> &nbsp;·&nbsp; </span>
    <span style="font-size:0.65rem; color:#0277bd; font-weight:600;">Matric:</span>
    <span style="font-size:0.72rem; color:#01579b; font-weight:700;"> 20231411152</span>
    <span style="font-size:0.65rem; color:#90caf9;"> &nbsp;·&nbsp; </span>
    <span style="font-size:0.65rem; color:#0277bd; font-weight:600;">Dept:</span>
    <span style="font-size:0.72rem; color:#01579b; font-weight:700;"> CYB</span>
</div>
""", unsafe_allow_html=True)
