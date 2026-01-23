import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Macro Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# HISTORICAL MACRO EVENT DATABASE
# (Can later be replaced with real dataset)
# -------------------------------
historical_events = pd.DataFrame([
    {
        "headline": "Russia invades Ukraine",
        "GPI": 0.9,
        "Economic": 0.6,
        "CrossBorder": 0.9,
        "AssetShock": 0.8,
        "severity": 0.9
    },
    {
        "headline": "US Federal Reserve raises interest rates sharply",
        "GPI": 0.2,
        "Economic": 0.9,
        "CrossBorder": 0.6,
        "AssetShock": 0.7,
        "severity": 0.7
    },
    {
        "headline": "Major bank collapses triggering financial crisis",
        "GPI": 0.4,
        "Economic": 1.0,
        "CrossBorder": 0.8,
        "AssetShock": 1.0,
        "severity": 1.0
    },
    {
        "headline": "Trade war escalates between US and China",
        "GPI": 0.5,
        "Economic": 0.8,
        "CrossBorder": 1.0,
        "AssetShock": 0.6,
        "severity": 0.8
    }
])

historical_events["embedding"] = historical_events["headline"].apply(
    lambda x: model.encode(x)
)

# -------------------------------
# PORTFOLIO (DEMO)
# -------------------------------
portfolio = pd.DataFrame({
    "Asset": ["US Equities", "International Equities", "Bonds", "Gold", "Cash"],
    "Weight": [0.45, 0.25, 0.20, 0.05, 0.05]
})

asset_sensitivity = {
    "US Equities": 0.8,
    "International Equities": 1.0,
    "Bonds": -0.5,
    "Gold": -0.9,
    "Cash": -0.2
}

# -------------------------------
# FUNCTIONS
# -------------------------------
def analyze_news(news_text):
    news_embedding = model.encode(news_text)
    similarities = cosine_similarity(
        [news_embedding],
        list(historical_events["embedding"])
    )[0]

    historical_events["similarity"] = similarities
    top_matches = historical_events.sort_values(
        by="similarity", ascending=False
    ).head(3)

    weighted_severity = np.average(
        top_matches["severity"],
        weights=top_matches["similarity"]
    )

    axes = {}
    for axis in ["GPI", "Economic", "CrossBorder", "AssetShock"]:
        axes[axis] = np.average(
            top_matches[axis],
            weights=top_matches["similarity"]
        )

    return weighted_severity, axes, top_matches


def rebalance_portfolio(severity):
    adjusted = portfolio.copy()
    adjusted["Adjustment"] = adjusted["Asset"].apply(
        lambda x: -severity * asset_sensitivity[x] * 0.05
    )
    adjusted["New Weight"] = adjusted["Weight"] + adjusted["Adjustment"]
    adjusted["New Weight"] = adjusted["New Weight"] / adjusted["New Weight"].sum()
    return adjusted


def explanation(asset, severity):
    if asset in ["Gold", "Bonds"]:
        return "Defensive allocation increased due to elevated macro risk."
    if asset == "Cash":
        return "Liquidity buffer raised to manage uncertainty."
    return "Risk exposure reduced due to elevated macro volatility."

# -------------------------------
# UI
# -------------------------------
st.title("📊 Macro Risk Intelligence Engine")
st.caption("Semantic macro risk detection • Portfolio impact • Institutional-style logic")

news_text = st.text_area(
    "Paste any news headline or article",
    height=120,
    placeholder="e.g. Political unrest intensifies after assassination attempt..."
)

if st.button("Analyze Impact"):
    severity, axes, matches = analyze_news(news_text)
    rebalanced = rebalance_portfolio(severity)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Macro Severity")
        st.metric("Severity Score", f"{severity:.2f}")

        st.subheader("📈 Macro Axes")
        st.json(axes)

    with col2:
        st.subheader("🧠 Closest Historical Analogs")
        st.dataframe(
            matches[["headline", "similarity", "severity"]],
            use_container_width=True
        )

    st.subheader("💼 Portfolio Rebalancing Logic")

    rebalanced["Explanation"] = rebalanced["Asset"].apply(
        lambda x: explanation(x, severity)
    )

    st.dataframe(
        rebalanced[["Asset", "Weight", "New Weight", "Explanation"]],
        use_container_width=True
    )

    st.info(
        "Rebalancing is driven by semantic similarity to historical macro events, "
        "not keywords. Severity reflects inferred market stress."
    )


