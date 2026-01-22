import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="MacroMoney",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MacroMoney")
st.caption(
    "Macro-aware portfolio decision support system based on historical analogs. "
    "This tool estimates expected market pressure — not guaranteed outcomes."
)

# ----------------------------------
# SIDEBAR – INVESTOR PROFILE
# ----------------------------------
st.sidebar.header("Investor Profile")

capital = st.sidebar.number_input(
    "Total Capital ($)",
    min_value=1000,
    value=100000,
    step=1000
)

horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short-term (<1 year)", "Medium-term (1–3 years)", "Long-term (3+ years)"]
)

risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance",
    ["Low", "Medium", "High"]
)

st.sidebar.subheader("Asset Allocation (%)")

assets = ["Equities", "Bonds", "Gold", "Commodities", "Crypto"]
weights = {}

for asset in assets:
    weights[asset] = st.sidebar.number_input(
        asset, min_value=0, max_value=100, value=20
    )

if sum(weights.values()) != 100:
    st.sidebar.error("Asset allocation must sum to 100%")

# ----------------------------------
# MACRO ARCHETYPE LIBRARY (CORE)
# ----------------------------------
ARCHETYPES = {
    "Political Shock": {
        "description": "Political instability, assassination attempts, leadership risk",
        "volatility": 0.9,
        "drawdown": 0.7,
        "contagion": 0.8,
        "betas": {"Equities": -0.6, "Bonds": 0.2, "Gold": 0.9, "Commodities": 0.3, "Crypto": -0.4}
    },
    "Commodity Price Breakout": {
        "description": "Gold/oil reaching extreme price levels",
        "volatility": 0.6,
        "drawdown": 0.3,
        "contagion": 0.6,
        "betas": {"Equities": -0.2, "Bonds": -0.1, "Gold": 0.8, "Commodities": 0.7, "Crypto": 0.1}
    },
    "Monetary Policy Shock": {
        "description": "Unexpected rate or inflation developments",
        "volatility": 0.7,
        "drawdown": 0.6,
        "contagion": 0.7,
        "betas": {"Equities": -0.4, "Bonds": -0.5, "Gold": 0.4, "Commodities": 0.5, "Crypto": -0.3}
    },
    "Corporate Earnings Shock": {
        "description": "Major earnings surprises from large firms",
        "volatility": 0.4,
        "drawdown": 0.2,
        "contagion": 0.3,
        "betas": {"Equities": 0.4, "Bonds": 0.0, "Gold": 0.0, "Commodities": 0.1, "Crypto": 0.2}
    },
    "Local / Noise": {
        "description": "Events with limited macro relevance",
        "volatility": 0.1,
        "drawdown": 0.1,
        "contagion": 0.1,
        "betas": {"Equities": 0, "Bonds": 0, "Gold": 0, "Commodities": 0, "Crypto": 0}
    }
}

# ----------------------------------
# SEMANTIC-LIKE MATCHING (DEMO LOGIC)
# ----------------------------------
def classify_event(news):
    news = news.lower()

    if any(x in news for x in ["assassination", "shot", "killed", "attack", "security incident"]):
        return "Political Shock"
    if any(x in news for x in ["gold", "oil", "commodity", "all-time high", "record high"]):
        return "Commodity Price Breakout"
    if any(x in news for x in ["interest rate", "inflation", "fed", "central bank"]):
        return "Monetary Policy Shock"
    if any(x in news for x in ["earnings", "profit", "loss", "quarter"]):
        return "Corporate Earnings Shock"

    return "Local / Noise"

# ----------------------------------
# SEVERITY INDEX (SYSTEM DETERMINED)
# ----------------------------------
def compute_severity(archetype):
    data = ARCHETYPES[archetype]
    severity = (
        0.5 * data["volatility"] +
        0.3 * data["drawdown"] +
        0.2 * data["contagion"]
    )
    return round(severity * 3, 2)

# ----------------------------------
# HORIZON DECAY
# ----------------------------------
HORIZON_DECAY = {
    "Short-term (<1 year)": 1.0,
    "Medium-term (1–3 years)": 0.7,
    "Long-term (3+ years)": 0.4
}

# ----------------------------------
# REBALANCING ENGINE
# ----------------------------------
def rebalance_portfolio(weights, archetype, severity, horizon, risk):
    betas = ARCHETYPES[archetype]["betas"]
    decay = HORIZON_DECAY[horizon]
    risk_scale = {"Low": 0.5, "Medium": 1.0, "High": 1.5}[risk]

    new_weights = weights.copy()

    for asset in new_weights:
        adjustment = betas[asset] * severity * decay * risk_scale * 2
        new_weights[asset] = max(0, new_weights[asset] + adjustment)

    total = sum(new_weights.values())
    return {k: round(v * 100 / total, 2) for k, v in new_weights.items()}

# ----------------------------------
# MAIN – NEWS INPUT
# ----------------------------------
st.subheader("📰 Macro Event Input")

news = st.text_area(
    "Enter a news headline or description",
    placeholder="Example: Gold prices hit all-time high amid global uncertainty"
)

if st.button("Analyze Macro Impact"):
    if not news.strip():
        st.warning("Please enter a news event.")
    else:
        archetype = classify_event(news)
        severity = compute_severity(archetype)
        effective_impact = severity * HORIZON_DECAY[horizon]

        st.divider()
        st.subheader("🧠 Macro Analysis")

        st.write(f"**Detected Archetype:** {archetype}")
        st.write(f"**Historical Context:** {ARCHETYPES[archetype]['description']}")
        st.write(f"**Severity Index:** {severity}")
        st.write(f"**Effective Impact (Horizon-adjusted):** {round(effective_impact,2)}")

        if archetype == "Local / Noise" or effective_impact < 1.2:
            st.info("This event does not justify portfolio rebalancing based on historical analogs.")
        else:
            st.success("Historical patterns suggest portfolio risk adjustment is justified.")

            before = pd.DataFrame(weights.items(), columns=["Asset", "Weight (%)"])
            after_weights = rebalance_portfolio(weights, archetype, severity, horizon, risk_tolerance)
            after = pd.DataFrame(after_weights.items(), columns=["Asset", "Weight (%)"])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Before")
                st.bar_chart(before.set_index("Asset"))
            with col2:
                st.subheader("After")
                st.bar_chart(after.set_index("Asset"))

            st.caption("Rebalancing reflects historical asset sensitivities and risk constraints.")
