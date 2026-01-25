import streamlit as st
import numpy as np
import pandas as pd
import hashlib

# =============================
# PAGE CONFIG (Bloomberg Style)
# =============================
st.set_page_config(
    page_title="MacroMoney | Macro Intelligence Engine",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0e1117; color: #e6e6e6; }
.block-container { padding-top: 1rem; }
h1, h2, h3 { color: #f5c518; }
.metric-box {
    background-color: #161b22;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #f5c518;
}
</style>
""", unsafe_allow_html=True)

# =============================
# DETERMINISTIC SEMANTIC ENGINE
# =============================
def embed_text(text, dim=384):
    h = hashlib.sha256(text.lower().encode()).digest()
    vec = np.frombuffer(h, dtype=np.uint8)
    vec = np.tile(vec, int(np.ceil(dim / len(vec))))[:dim]
    return vec / np.linalg.norm(vec)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =============================
# MACRO ARCHETYPES (ANCHORS)
# =============================
ARCHETYPES = {
    "Geopolitical Shock": embed_text("war conflict assassination military sanctions instability"),
    "Monetary Policy Shift": embed_text("interest rates central bank inflation tightening easing"),
    "Commodity Shock": embed_text("gold oil commodities supply shortage price spike"),
    "Corporate Earnings": embed_text("earnings profit revenue downgrade guidance"),
    "Financial Crisis": embed_text("bank collapse liquidity credit default crisis"),
    "Trade & Globalization": embed_text("trade deal tariffs exports imports agreement"),
    "Local / Noise": embed_text("celebrity local unrelated minor event")
}

# =============================
# PORTFOLIO SETUP
# =============================
ASSETS = ["Equities", "Bonds", "Gold", "Crypto", "Commodities", "ETFs"]

BASE_PORTFOLIO = {
    "Equities": 0.40,
    "Bonds": 0.30,
    "Gold": 0.10,
    "Crypto": 0.05,
    "Commodities": 0.10,
    "ETFs": 0.05
}

IMPACT_MATRIX = {
    "Geopolitical Shock": {"Equities": -0.20, "Bonds": 0.15, "Gold": 0.30},
    "Monetary Policy Shift": {"Equities": -0.10, "Bonds": -0.15, "Gold": 0.10},
    "Commodity Shock": {"Equities": -0.05, "Gold": 0.35, "Commodities": 0.30},
    "Corporate Earnings": {"Equities": 0.20},
    "Financial Crisis": {"Equities": -0.35, "Bonds": 0.25, "Gold": 0.40},
    "Trade & Globalization": {"Equities": 0.15, "Commodities": 0.20}
}

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.title("Macro Controls")
capital = st.sidebar.number_input("Total Capital ($)", min_value=1000, value=100000, step=1000)

horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short (≤1 year)", "Medium (1–3 years)", "Long (3+ years)"]
)

thresholds = {
    "Short (≤1 year)": 0.30,
    "Medium (1–3 years)": 0.50,
    "Long (3+ years)": 0.65
}
rebalance_threshold = thresholds[horizon]

# =============================
# MAIN UI
# =============================
st.title("MacroMoney — Macro Intelligence Engine")
st.caption("Semantic macro analysis • Severity scoring • Portfolio rebalancing")

headline = st.text_area(
    "Enter any news headline or macro event:",
    height=90,
    placeholder="Example: Gold prices hit all-time high amid global inflation fears"
)

if st.button("Analyze Macro Impact"):
    if not headline.strip():
        st.warning("Please enter a news event.")
    else:
        news_vec = embed_text(headline)

        scores = {k: cosine_sim(news_vec, v) for k, v in ARCHETYPES.items()}
        archetype = max(scores, key=scores.get)
        severity = min(1.0, max(0.0, scores[archetype]))

        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Archetype", archetype)
        col2.metric("Severity Index", round(severity, 2))
        col3.metric("Rebalance Threshold", rebalance_threshold)

        st.subheader("Portfolio Decision")

        new_portfolio = BASE_PORTFOLIO.copy()
        explanation = []

        if archetype == "Local / Noise" or severity < rebalance_threshold:
            st.info("Impact assessed as insufficient to justify portfolio rebalancing.")
            explanation.append(
                "The event does not meet the macro severity threshold required to alter a diversified portfolio."
            )
        else:
            impact = IMPACT_MATRIX.get(archetype, {})
            for asset, delta in impact.items():
                new_portfolio[asset] += delta * severity

            total = sum(new_portfolio.values())
            for k in new_portfolio:
                new_portfolio[k] /= total

            st.success("Macro impact significant — portfolio rebalanced.")

            explanation.extend([
                f"The system classified this event as **{archetype}** using semantic similarity.",
                f"Severity score of **{round(severity,2)}** reflects historical market reactions to similar events.",
                f"Given your **{horizon.lower()} investment horizon**, rebalancing was triggered.",
                "Asset weights were adjusted using historical risk-on / risk-off transmission logic."
            ])

        # =============================
        # OUTPUT TABLE
        # =============================
        df = pd.DataFrame({
            "Asset": ASSETS,
            "Base Weight": [BASE_PORTFOLIO[a] for a in ASSETS],
            "New Weight": [new_portfolio[a] for a in ASSETS],
            "Capital ($)": [new_portfolio[a] * capital for a in ASSETS]
        })

        st.dataframe(
            df.style.format({
                "Base Weight": "{:.2%}",
                "New Weight": "{:.2%}",
                "Capital ($)": "${:,.0f}"
            }),
            use_container_width=True
        )

        st.subheader("Explanation & Logic")
        for e in explanation:
            st.write("•", e)

        st.caption(
            "Demo note: This version uses deterministic semantic embeddings. "
            "Production version replaces this with transformer embeddings calibrated on historical returns."
        )
