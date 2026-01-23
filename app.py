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
    "Estimates expected market pressure — not guaranteed outcomes."
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
    weights[asset] = st.sidebar.number_input(asset, min_value=0, max_value=100, value=20)

if sum(weights.values()) != 100:
    st.sidebar.error("Asset allocation must sum to 100%")

# ----------------------------------
# MACRO ARCHETYPE LIBRARY
# ----------------------------------
ARCHETYPES = {
    "Political Shock": {
        "description": "Political instability, assassination attempts, leadership risk",
        "betas": {"Equities": -0.6, "Bonds": 0.2, "Gold": 0.9, "Commodities": 0.3, "Crypto": -0.4}
    },
    "Commodity Price Breakout": {
        "description": "Gold/oil reaching extreme price levels",
        "betas": {"Equities": -0.2, "Bonds": -0.1, "Gold": 0.8, "Commodities": 0.7, "Crypto": 0.1}
    },
    "Monetary Policy Shock": {
        "description": "Unexpected rate or inflation developments",
        "betas": {"Equities": -0.4, "Bonds": -0.5, "Gold": 0.4, "Commodities": 0.5, "Crypto": -0.3}
    },
    "Corporate Earnings Shock": {
        "description": "Major earnings surprises from large firms",
        "betas": {"Equities": 0.4, "Bonds": 0.0, "Gold": 0.0, "Commodities": 0.1, "Crypto": 0.2}
    },
    "Trade / Commodity Deal": {
        "description": "Major cross-border trade agreements affecting commodities",
        "betas": {"Equities": 0.2, "Bonds": 0.0, "Gold": 0.3, "Commodities": 0.5, "Crypto": 0.1}
    },
    "Geopolitical Escalation": {
        "description": "Wars, sanctions, global military conflicts",
        "betas": {"Equities": -0.8, "Bonds": 0.3, "Gold": 1.0, "Commodities": 0.6, "Crypto": -0.5}
    },
    "Local / Noise": {
        "description": "Events with limited macro relevance",
        "betas": {"Equities": 0, "Bonds": 0, "Gold": 0, "Commodities": 0, "Crypto": 0}
    }
}

# ----------------------------------
# HORIZON DECAY
# ----------------------------------
HORIZON_DECAY = {
    "Short-term (<1 year)": 1.0,
    "Medium-term (1–3 years)": 0.7,
    "Long-term (3+ years)": 0.4
}

# ----------------------------------
# MACRO AXIS SCORING (DEMO RULES)
# ----------------------------------
def macro_axis_scores(news):
    news_lower = news.lower()
    axes = {
        "Geopolitical Intensity": 0,
        "Economic Scale": 0,
        "Cross-Border Impact": 0,
        "Asset Transmission": 0
    }

    # Political / war events
    if any(x in news_lower for x in ["war", "attack", "assassination", "conflict", "sanctions"]):
        axes["Geopolitical Intensity"] = 5
        axes["Economic Scale"] = 4
        axes["Cross-Border Impact"] = 5
        axes["Asset Transmission"] = 5
    # Trade / commodity deals
    elif any(x in news_lower for x in ["deal", "trade", "billion", "metal", "commodity"]):
        axes["Geopolitical Intensity"] = 2
        axes["Economic Scale"] = 4
        axes["Cross-Border Impact"] = 4
        axes["Asset Transmission"] = 4
    # Corporate earnings
    elif any(x in news_lower for x in ["earnings", "profit", "loss", "quarter"]):
        axes["Geopolitical Intensity"] = 1
        axes["Economic Scale"] = 3
        axes["Cross-Border Impact"] = 2
        axes["Asset Transmission"] = 3
    # Monetary policy
    elif any(x in news_lower for x in ["fed", "inflation", "interest rate", "central bank"]):
        axes["Geopolitical Intensity"] = 1
        axes["Economic Scale"] = 4
        axes["Cross-Border Impact"] = 4
        axes["Asset Transmission"] = 4
    else:
        axes["Geopolitical Intensity"] = 0
        axes["Economic Scale"] = 0
        axes["Cross-Border Impact"] = 0
        axes["Asset Transmission"] = 0

    return axes

# ----------------------------------
# CLASSIFY ARCHETYPE BASED ON AXIS
# ----------------------------------
def classify_archetype(axes):
    gpi = axes["Geopolitical Intensity"]
    es = axes["Economic Scale"]
    cbi = axes["Cross-Border Impact"]
    at = axes["Asset Transmission"]

    if gpi >= 4:
        return "Geopolitical Escalation"
    elif es >=4 and at >=4 and gpi>=2:
        return "Trade / Commodity Deal"
    elif gpi<=1 and es>=3 and cbi>=2 and at>=3:
        return "Corporate Earnings Shock"
    elif es>=4 and gpi<=2 and at>=3:
        return "Monetary Policy Shock"
    elif gpi==0 and es==0:
        return "Local / Noise"
    else:
        return "Political Shock"

# ----------------------------------
# SEVERITY CALCULATION
# ----------------------------------
def compute_severity(axes):
    weights = {"Geopolitical Intensity": 0.35,
               "Economic Scale": 0.30,
               "Cross-Border Impact": 0.20,
               "Asset Transmission": 0.15}
    severity = sum(axes[k]*v for k,v in weights.items())
    return round(severity,2)

# ----------------------------------
# PORTFOLIO REBALANCING
# ----------------------------------
def rebalance_portfolio(weights, archetype, severity, horizon, risk):
    betas = ARCHETYPES[archetype]["betas"]
    decay = HORIZON_DECAY[horizon]
    risk_scale = {"Low": 0.5, "Medium":1.0, "High":1.5}[risk]

    new_weights = weights.copy()
    for asset in new_weights:
        adjustment = betas[asset]*severity*decay*risk_scale*2
        new_weights[asset] = max(0,new_weights[asset]+adjustment)
    total = sum(new_weights.values())
    return {k: round(v*100/total,2) for k,v in new_weights.items()}

# ----------------------------------
# MAIN APP
# ----------------------------------
st.subheader("📰 Macro Event Input")
news = st.text_area("Enter a news headline or description",
                    placeholder="Example: Gold prices hit all-time high amid global uncertainty")

if st.button("Analyze Macro Impact"):
    if not news.strip():
        st.warning("Please enter a news event.")
    else:
        axes = macro_axis_scores(news)
        archetype = classify_archetype(axes)
        severity = compute_severity(axes)
        effective_impact = severity*HORIZON_DECAY[horizon]

        st.divider()
        st.subheader("🧠 Macro Analysis")

        st.write(f"**Detected Archetype:** {archetype}")
        st.write(f"**Historical Context:** {ARCHETYPES[archetype]['description']}")
        st.write(f"**Severity Index:** {severity}")
        st.write(f"**Effective Impact (Horizon-adjusted):** {round(effective_impact,2)}")

        st.write("**Macro Axis Scores:**")
        st.table(pd.DataFrame.from_dict(axes, orient="index", columns=["Score"]))

        if archetype=="Local / Noise" or effective_impact < 1.5:
            st.info("This event does not justify portfolio rebalancing based on historical analogs.")
        else:
            st.success("Historical patterns suggest portfolio risk adjustment is justified.")

            before = pd.DataFrame(weights.items(), columns=["Asset","Weight (%)"])
            after_weights = rebalance_portfolio(weights, archetype, severity, horizon, risk_tolerance)
            after = pd.DataFrame(after_weights.items(), columns=["Asset","Weight (%)"])

            col1,col2 = st.columns(2)
            with col1:
                st.subheader("Before")
                st.bar_chart(before.set_index("Asset"))
            with col2:
                st.subheader("After")
                st.bar_chart(after.set_index("Asset"))

            st.caption("Rebalancing reflects historical asset sensitivities, severity, and risk constraints.")

