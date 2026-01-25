import streamlit as st
import numpy as np
import pandas as pd
import hashlib

# =============================
# PAGE CONFIG
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
.section {
    background-color: #161b22;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SEMANTIC ENGINE (STABLE DEMO)
# =============================
def embed_text(text, dim=384):
    h = hashlib.sha256(text.lower().encode()).digest()
    vec = np.frombuffer(h, dtype=np.uint8)
    vec = np.tile(vec, int(np.ceil(dim / len(vec))))[:dim]
    return vec / np.linalg.norm(vec)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =============================
# MACRO ARCHETYPES
# =============================
ARCHETYPES = {
    "Geopolitical Shock": embed_text("war assassination conflict sanctions military instability"),
    "Monetary Policy Shift": embed_text("interest rates central bank inflation tightening easing"),
    "Commodity Shock": embed_text("gold oil commodities supply shock price spike"),
    "Corporate Earnings": embed_text("earnings profit revenue downgrade guidance"),
    "Financial Crisis": embed_text("bank collapse liquidity credit crisis"),
    "Trade & Globalization": embed_text("trade deal tariffs exports imports agreement"),
    "Local / Noise": embed_text("celebrity local unrelated minor event")
}

ARCHETYPE_EXPLANATION = {
    "Geopolitical Shock": "Geopolitical shocks increase uncertainty and risk aversion across global markets.",
    "Monetary Policy Shift": "Changes in monetary policy directly affect liquidity, borrowing costs, and asset valuations.",
    "Commodity Shock": "Commodity shocks signal inflationary pressure or supply constraints in the global economy.",
    "Corporate Earnings": "Earnings events primarily impact equity valuations and investor sentiment.",
    "Financial Crisis": "Financial crises trigger systemic risk and capital flight toward safe-haven assets.",
    "Trade & Globalization": "Trade developments influence growth expectations and cross-border capital flows.",
    "Local / Noise": "This event lacks sufficient macroeconomic relevance to impact diversified portfolios."
}

# =============================
# PORTFOLIO & IMPACT MODEL
# =============================
ASSETS = ["Equities", "Bonds", "Gold", "Crypto", "Commodities", "ETFs"]

IMPACT_MATRIX = {
    "Geopolitical Shock": {"Equities": -0.20, "Bonds": 0.15, "Gold": 0.30},
    "Monetary Policy Shift": {"Equities": -0.10, "Bonds": -0.15, "Gold": 0.10},
    "Commodity Shock": {"Equities": -0.05, "Gold": 0.35, "Commodities": 0.30},
    "Corporate Earnings": {"Equities": 0.20},
    "Financial Crisis": {"Equities": -0.35, "Bonds": 0.25, "Gold": 0.40},
    "Trade & Globalization": {"Equities": 0.15, "Commodities": 0.20}
}

# =============================
# SIDEBAR — USER PORTFOLIO
# =============================
st.sidebar.title("Portfolio Setup")

capital = st.sidebar.number_input(
    "Total Capital ($)", min_value=1000, value=100000, step=1000
)

st.sidebar.markdown("### Initial Allocation (%)")

weights = {}
total_weight = 0

for asset in ASSETS:
    w = st.sidebar.slider(asset, 0, 100, 100 // len(ASSETS))
    weights[asset] = w / 100
    total_weight += w

if total_weight != 100:
    st.sidebar.error("Total allocation must equal 100%")
    st.stop()

st.sidebar.markdown("### Investment Horizon")
horizon = st.sidebar.selectbox(
    "", ["Short (≤1 year)", "Medium (1–3 years)", "Long (3+ years)"]
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
st.caption("Macro reasoning • Severity scoring • Portfolio intelligence")

headline = st.text_area(
    "Enter any news headline or macro event:",
    height=90
)

if st.button("Analyze Macro Impact"):
    if not headline.strip():
        st.warning("Please enter a news event.")
    else:
        news_vec = embed_text(headline)
        scores = {k: cosine_sim(news_vec, v) for k, v in ARCHETYPES.items()}
        archetype = max(scores, key=scores.get)
        severity = min(1.0, max(0.0, scores[archetype]))

        st.markdown("## Macro Assessment")

        c1, c2, c3 = st.columns(3)
        c1.metric("Detected Archetype", archetype)
        c2.metric("Severity Index", round(severity, 2))
        c3.metric("Rebalance Threshold", rebalance_threshold)

        new_portfolio = weights.copy()
        explanation = []

        explanation.append(f"**Macro Interpretation:** {ARCHETYPE_EXPLANATION[archetype]}")

        if archetype == "Local / Noise" or severity < rebalance_threshold:
            explanation.append(
                "The severity of this event is not high enough to justify portfolio changes "
                "given your investment horizon."
            )
            st.info("No portfolio rebalancing triggered.")
        else:
            impact = IMPACT_MATRIX.get(archetype, {})
            for asset, delta in impact.items():
                new_portfolio[asset] += delta * severity

            total = sum(new_portfolio.values())
            for k in new_portfolio:
                new_portfolio[k] /= total

            explanation.append(
                "Historically, markets respond to this type of event by reallocating capital "
                "toward assets that offer protection or benefit from the macro shift."
            )

            for asset, delta in impact.items():
                direction = "increased" if delta > 0 else "reduced"
                explanation.append(
                    f"Exposure to **{asset}** was {direction} due to its historical behavior during similar events."
                )

            st.success("Portfolio rebalanced based on macro severity.")

        # =============================
        # PORTFOLIO TABLE
        # =============================
        df = pd.DataFrame({
            "Asset": ASSETS,
            "Initial Weight": [weights[a] for a in ASSETS],
            "New Weight": [new_portfolio[a] for a in ASSETS],
            "Capital ($)": [new_portfolio[a] * capital for a in ASSETS]
        })

        st.markdown("## Portfolio Allocation")
        st.dataframe(
            df.style.format({
                "Initial Weight": "{:.2%}",
                "New Weight": "{:.2%}",
                "Capital ($)": "${:,.0f}"
            }),
            use_container_width=True
        )

        # =============================
        # EXPLANATION PANEL
        # =============================
        st.markdown("## Explanation & Market Logic")
        for e in explanation:
            st.write("•", e)

        st.caption(
            "Demo note: Semantic engine is deterministic for stability. "
            "Production version will use transformer embeddings calibrated on historical returns."
        )

