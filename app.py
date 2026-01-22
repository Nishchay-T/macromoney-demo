import streamlit as st
import pandas as pd

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="MacroMoney Demo",
    layout="wide",
    page_icon="📊"
)

st.title("📊 MacroMoney – Demo Version")
st.caption("Macro-aware portfolio insights | Demo model")

# -----------------------------
# SIDEBAR – USER PROFILE
# -----------------------------
st.sidebar.header("Investor Profile")

capital = st.sidebar.number_input(
    "Total Capital ($)",
    min_value=1000,
    step=1000,
    value=100000
)

horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short-term (<1 year)", "Medium-term (1–3 years)", "Long-term (3+ years)"]
)

risk = st.sidebar.selectbox(
    "Risk Tolerance",
    ["Low", "Medium", "High"]
)

st.sidebar.subheader("Asset Allocation (%)")

assets = ["Equities", "Bonds", "Gold", "Crypto", "Commodities", "ETFs"]
weights = {}

for asset in assets:
    weights[asset] = st.sidebar.number_input(
        asset,
        min_value=0,
        max_value=100,
        value=100 // len(assets)
    )

if sum(weights.values()) != 100:
    st.sidebar.error("⚠️ Asset weights must sum to 100%")

# -----------------------------
# MAIN – NEWS INPUT
# -----------------------------
st.subheader("📰 Macro Event Analysis")

news = st.text_area(
    "Enter News Headline",
    placeholder="Example: Gold prices hit all-time high amid global uncertainty"
)

severity = st.selectbox(
    "Event Severity (manual for demo)",
    ["Low", "Medium", "High"]
)

# -----------------------------
# MACRO THEME ENGINE
# -----------------------------
THEMES = {
    "Geopolitics": ["war", "attack", "assassination", "conflict", "military"],
    "Inflation / Rates": ["inflation", "interest rate", "fed", "central bank"],
    "Commodities Shock": ["gold", "oil", "commodity", "all-time high", "prices hit"],
    "Corporate Earnings": ["earnings", "profit", "loss", "quarter"],
    "Technology": ["tech", "ai", "semiconductor", "software"]
}

def detect_theme(text):
    text = text.lower()
    for theme, keywords in THEMES.items():
        for kw in keywords:
            if kw in text:
                return theme
    return "Local / Irrelevant News"

# -----------------------------
# IMPACT LOGIC
# -----------------------------
SEVERITY_SCORE = {"Low": 1, "Medium": 2, "High": 3}

HORIZON_THRESHOLD = {
    "Short-term (<1 year)": 1,
    "Medium-term (1–3 years)": 2,
    "Long-term (3+ years)": 3
}

def should_rebalance(severity, horizon):
    return SEVERITY_SCORE[severity] >= HORIZON_THRESHOLD[horizon]

# -----------------------------
# ANALYSIS BUTTON
# -----------------------------
if st.button("Analyze Macro Impact"):
    if not news.strip():
        st.warning("Please enter a news headline.")
    else:
        theme = detect_theme(news)

        st.divider()
        st.subheader("🧠 Analysis Result")

        st.write(f"**Detected Theme:** {theme}")
        st.write(f"**Event Severity:** {severity}")
        st.write(f"**Investment Horizon:** {horizon}")

        if theme == "Local / Irrelevant News":
            st.info("This event is unlikely to have macro-level portfolio impact.")
        else:
            if should_rebalance(severity, horizon):
                st.success("📈 Portfolio rebalancing is justified given your horizon.")

                df = pd.DataFrame({
                    "Asset": list(weights.keys()),
                    "Allocation (%)": list(weights.values())
                })

                st.subheader("Current Portfolio Allocation")
                st.bar_chart(df.set_index("Asset"))
            else:
                st.warning(
                    "Impact exists, but not strong enough to rebalance given your investment horizon."
                )
