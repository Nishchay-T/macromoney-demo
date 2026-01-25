import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="MacroMoney",
    page_icon="📊",
    layout="wide"
)

HF_API_KEY = st.secrets["HF_API_KEY"]
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------ STYLES ------------------
st.markdown("""
<style>
body { background-color: #0E1117; }
.metric-box {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ FUNCTIONS ------------------
def embed_text(text):
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text}
    )
    return np.mean(response.json(), axis=0)

def severity_from_axes(axes):
    return np.clip(np.mean(axes), 0, 1)

def rebalance_portfolio(base_weights, severity, horizon):
    horizon_factor = np.clip(1 / horizon, 0.1, 1)
    adjustment = severity * horizon_factor

    shifts = {
        "Equities": -0.3 * adjustment,
        "Bonds": 0.2 * adjustment,
        "Gold": 0.25 * adjustment,
        "Crypto": -0.1 * adjustment,
        "Commodities": 0.15 * adjustment
    }

    new_weights = {}
    for k in base_weights:
        new_weights[k] = max(base_weights[k] + shifts.get(k, 0), 0)

    total = sum(new_weights.values())
    return {k: v / total for k, v in new_weights.items()}

# ------------------ HISTORICAL EVENT LIBRARY ------------------
historical_events = [
    {
        "headline": "Russia invades Ukraine",
        "axes": [0.95, 0.85, 0.9, 0.8],
        "impact": "Geopolitical Shock"
    },
    {
        "headline": "US Federal Reserve raises interest rates aggressively",
        "axes": [0.4, 0.9, 0.6, 0.85],
        "impact": "Monetary Tightening"
    },
    {
        "headline": "Global financial crisis triggered by banking collapse",
        "axes": [0.6, 0.95, 0.85, 0.95],
        "impact": "Systemic Financial Risk"
    },
    {
        "headline": "Gold prices hit all-time high amid uncertainty",
        "axes": [0.3, 0.6, 0.5, 0.8],
        "impact": "Safe Haven Demand"
    },
    {
        "headline": "Assassination attempt on US President",
        "axes": [0.9, 0.7, 0.8, 0.6],
        "impact": "Political Instability"
    }
]

for e in historical_events:
    e["embedding"] = embed_text(e["headline"])

# ------------------ UI ------------------
st.title("📊 MacroMoney")
st.caption("AI-driven macro risk interpretation & portfolio intelligence")

left, right = st.columns([1.2, 1])

with left:
    news = st.text_area("📰 Enter News / Headline", height=120)
    horizon = st.slider("Investment Horizon (Years)", 1, 20, 5)

with right:
    st.subheader("Initial Portfolio Weights")
    equities = st.slider("Equities", 0.0, 1.0, 0.4)
    bonds = st.slider("Bonds", 0.0, 1.0, 0.25)
    gold = st.slider("Gold", 0.0, 1.0, 0.15)
    crypto = st.slider("Crypto", 0.0, 1.0, 0.1)
    commodities = st.slider("Commodities", 0.0, 1.0, 0.1)

    base_weights = {
        "Equities": equities,
        "Bonds": bonds,
        "Gold": gold,
        "Crypto": crypto,
        "Commodities": commodities
    }

# ------------------ ANALYSIS ------------------
if st.button("Analyze Macro Impact") and news.strip():
    news_embedding = embed_text(news)

    sims = []
    for e in historical_events:
        sim = cosine_similarity(
            [news_embedding], [e["embedding"]]
        )[0][0]
        sims.append(sim)

    top_idx = np.argmax(sims)
    analog = historical_events[top_idx]

    axes = analog["axes"]
    severity = severity_from_axes(axes)
    new_portfolio = rebalance_portfolio(base_weights, severity, horizon)

    # ------------------ OUTPUT ------------------
    st.divider()
    st.subheader("🧠 Macro Interpretation")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Archetype", analog["impact"])
    col2.metric("Severity Index", round(severity, 2))
    col3.metric("Closest Historical Analog", analog["headline"])

    st.subheader("📊 Macro Axis Scores")
    axis_df = pd.DataFrame({
        "Axis": ["Geopolitical", "Economic Scale", "Cross-Border Impact", "Asset Transmission"],
        "Score": axes
    })
    st.bar_chart(axis_df.set_index("Axis"))

    st.subheader("📈 Portfolio Rebalancing")
    result_df = pd.DataFrame({
        "Asset": base_weights.keys(),
        "Old Weight": base_weights.values(),
        "New Weight": new_portfolio.values()
    })
    st.dataframe(result_df, use_container_width=True)

    st.subheader("📝 Explanation")
    st.write(
        f"""
        The system compared the news to historical macro events using semantic embeddings.
        The closest analog was **{analog['headline']}**, classified as **{analog['impact']}**.

        Given a **severity of {round(severity,2)}** and an investment horizon of **{horizon} years**,
        the portfolio was adjusted to reduce risk-sensitive assets and increase defensive exposure.

        Rebalancing magnitude is horizon-adjusted and severity-weighted.
        """
    )



