import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from core.monte_carlo_basic import monte_carlo_option_pricing
from core.variance_reduction import monte_carlo_antithetic, monte_carlo_control_variate
from core.black_scholes import black_scholes_call, black_scholes_put
from core.monte_carlo_exotic import (
    monte_carlo_asian_arithmetic,
    monte_carlo_asian_geometric,
    asian_geometric_call_analytic,
)
from core.greeks import monte_carlo_delta, monte_carlo_gamma_crn


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="MC OPTION PRICING // TERMINAL",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BRUTALIST STYLE
# ============================================================
BRUTALIST_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Space+Grotesk:wght@500;700;800&display=swap');

:root {
  --bg: #f4f1ea;
  --ink: #0a0a0a;
  --paper: #ffffff;
  --accent: #ff3c00;
  --accent-2: #ffe600;
  --muted: #6b6b6b;
  --rule: #0a0a0a;
}

html, body, [class*="css"], .stApp, .stMarkdown, .block-container {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace !important;
  color: var(--ink) !important;
}

.stApp {
  background: var(--bg) !important;
  background-image:
    linear-gradient(var(--ink) 1px, transparent 1px),
    linear-gradient(90deg, var(--ink) 1px, transparent 1px);
  background-size: 48px 48px;
  background-position: -1px -1px;
  background-blend-mode: normal;
}
.stApp::before {
  content: "";
  position: fixed; inset: 0;
  background: var(--bg);
  opacity: 0.92;
  pointer-events: none;
  z-index: 0;
}
.block-container { position: relative; z-index: 1; padding-top: 1.2rem !important; }

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  text-transform: uppercase;
  color: var(--ink) !important;
}

hr { border: none; border-top: 2px solid var(--ink); margin: 0.6rem 0 1.2rem 0; }

/* Header banner */
.brutal-banner {
  border: 3px solid var(--ink);
  background: var(--paper);
  padding: 18px 22px;
  box-shadow: 8px 8px 0 var(--ink);
  margin-bottom: 28px;
  position: relative;
}
.brutal-banner .tag {
  display: inline-block;
  background: var(--ink);
  color: var(--paper);
  padding: 2px 8px;
  font-size: 11px;
  letter-spacing: 0.2em;
  margin-bottom: 8px;
}
.brutal-banner h1 {
  font-size: 40px !important;
  margin: 0 !important;
  line-height: 1 !important;
}
.brutal-banner .sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.05em;
  margin-top: 10px;
  color: var(--muted);
  text-transform: uppercase;
}
.brutal-banner .corner {
  position: absolute; top: -14px; right: 14px;
  background: var(--accent); color: var(--paper);
  padding: 4px 10px; font-size: 11px; letter-spacing: 0.2em;
  border: 2px solid var(--ink);
}

/* Cards / panels */
.brutal-card {
  border: 3px solid var(--ink);
  background: var(--paper);
  padding: 18px 20px;
  box-shadow: 6px 6px 0 var(--ink);
  margin-bottom: 18px;
}
.brutal-card.accent { background: var(--accent-2); }
.brutal-card.dark   { background: var(--ink); color: var(--paper); }
.brutal-card.dark *, .brutal-card.dark h1, .brutal-card.dark h2 { color: var(--paper) !important; }

.brutal-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
  display: block;
}

.brutal-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 42px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput input,
.stTextInput input {
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  background: var(--paper) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 600 !important;
  box-shadow: 3px 3px 0 var(--ink) !important;
}
.stNumberInput button {
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  background: var(--paper) !important;
  color: var(--ink) !important;
}
label, .stSelectbox label, .stNumberInput label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
}

/* Button */
.stButton > button {
  width: 100%;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  background: var(--accent) !important;
  color: var(--paper) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 800 !important;
  text-transform: uppercase;
  letter-spacing: 0.15em !important;
  padding: 14px 20px !important;
  box-shadow: 6px 6px 0 var(--ink) !important;
  transition: transform 0.05s linear, box-shadow 0.05s linear !important;
}
.stButton > button:hover {
  transform: translate(2px, 2px);
  box-shadow: 4px 4px 0 var(--ink) !important;
  background: var(--ink) !important;
  color: var(--accent-2) !important;
}
.stButton > button:active {
  transform: translate(6px, 6px);
  box-shadow: 0 0 0 var(--ink) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 3px solid var(--ink);
}
[data-testid="stSidebar"] * { color: var(--paper) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput input {
  background: var(--paper) !important;
  color: var(--ink) !important;
  box-shadow: 3px 3px 0 var(--accent) !important;
}
[data-testid="stSidebar"] label {
  color: var(--accent-2) !important;
}

/* Alerts */
div[data-testid="stAlert"] {
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  box-shadow: 5px 5px 0 var(--ink) !important;
}

/* Expander */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  background: var(--paper) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 0.1em !important;
  box-shadow: 4px 4px 0 var(--ink) !important;
}
[data-testid="stExpander"] {
  border: 0 !important;
  background: transparent !important;
}
[data-testid="stExpander"] > details {
  border: 0 !important;
  background: transparent !important;
}

/* Footer ticker */
.brutal-foot {
  border-top: 2px solid var(--ink);
  padding-top: 10px;
  margin-top: 36px;
  font-size: 11px;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(BRUTALIST_CSS, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="brutal-banner">
      <div class="corner">v.01 // LIVE</div>
      <span class="tag">QUANT // MONTE CARLO ENGINE</span>
      <h1>OPTION&nbsp;PRICING&nbsp;TERMINAL</h1>
      <div class="sub">[ EUROPEAN ] [ ASIAN ] [ VARIANCE REDUCTION ] [ GREEKS ]</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR  ::  PARAMETERS
# ============================================================
with st.sidebar:
    st.markdown("### // PARAMETERS")
    st.markdown("<hr/>", unsafe_allow_html=True)

    option_type = st.selectbox("Option Type", ["call", "put"])
    method = st.selectbox(
        "Pricing Method",
        [
            "Basic Monte Carlo",
            "Antithetic Variates",
            "Control Variate",
            "Asian Arithmetic",
            "Asian Geometric (MC)",
            "Asian Geometric (Analytic)",
        ],
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### // MARKET")

    S0 = st.number_input("Spot Price  S_0", value=100.0, step=1.0, format="%.4f")
    K = st.number_input("Strike Price  K", value=100.0, step=1.0, format="%.4f")
    T = st.number_input("Maturity  T  (years)", value=1.0, step=0.25, format="%.4f")
    r = st.number_input("Risk-free Rate  r", value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility  sigma", value=0.20, step=0.01, format="%.4f")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### // ENGINE")
    num_sim = st.number_input(
        "Simulations  N", min_value=1000, value=100_000, step=1000
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    run = st.button("[ RUN PRICING ]")


# ============================================================
# MAIN  ::  SUMMARY GRID
# ============================================================
left, mid, right = st.columns([1.2, 1, 1])

with left:
    st.markdown(
        f"""
        <div class="brutal-card">
          <span class="brutal-label">// METHOD</span>
          <div class="brutal-value" style="font-size:24px;">{method.upper()}</div>
          <div style="margin-top:14px;">
            <span class="brutal-label">Type</span>
            <div style="font-weight:700;">{option_type.upper()}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mid:
    st.markdown(
        f"""
        <div class="brutal-card accent">
          <span class="brutal-label">// MONEYNESS  S_0/K</span>
          <div class="brutal-value">{(S0 / K):.3f}</div>
          <div style="margin-top:10px; font-size:11px; letter-spacing:0.2em;">
            { 'IN-THE-MONEY' if (option_type=='call' and S0>K) or (option_type=='put' and S0<K) else 'OUT-OF-THE-MONEY' if (option_type=='call' and S0<K) or (option_type=='put' and S0>K) else 'AT-THE-MONEY' }
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="brutal-card dark">
          <span class="brutal-label" style="color:#bdbdbd;">// PATHS</span>
          <div class="brutal-value">{int(num_sim):,}</div>
          <div style="margin-top:10px; font-size:11px; letter-spacing:0.2em;">
            sigma = {sigma:.2%}  //  T = {T:.2f}y  //  r = {r:.2%}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PRICING EXECUTION
# ============================================================
def compute_price(method, option_type, S0, K, T, r, sigma, num_sim):
    n = int(num_sim)
    if method == "Basic Monte Carlo":
        return monte_carlo_option_pricing(S0, K, T, r, sigma, n, option_type)
    if method == "Antithetic Variates":
        return monte_carlo_antithetic(S0, K, T, r, sigma, n, option_type)
    if method == "Control Variate":
        return monte_carlo_control_variate(S0, K, T, r, sigma, n, option_type)
    if method == "Asian Arithmetic":
        return monte_carlo_asian_arithmetic(S0, K, T, r, sigma, n, option_type=option_type)
    if method == "Asian Geometric (MC)":
        return monte_carlo_asian_geometric(S0, K, T, r, sigma, n, option_type=option_type)
    if method == "Asian Geometric (Analytic)":
        if option_type == "call":
            return asian_geometric_call_analytic(S0, K, T, r, sigma)
        return None
    return None


if run:
    with st.spinner("// SIMULATING ..."):
        t0 = time.perf_counter()
        if method == "Asian Geometric (Analytic)" and option_type == "put":
            st.markdown(
                """
                <div class="brutal-card" style="background:var(--accent); color:#fff;">
                  <span class="brutal-label" style="color:#fff;">// ERROR  [×]</span>
                  <div style="font-weight:700; font-size:18px;">
                    ANALYTIC FORMULA AVAILABLE FOR GEOMETRIC <u>CALL</u> ONLY
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            price = None
        else:
            price = compute_price(method, option_type, S0, K, T, r, sigma, num_sim)
        elapsed = time.perf_counter() - t0

    if price is not None:
        # Reference Black-Scholes price for European types
        bs_ref = None
        if method in ("Basic Monte Carlo", "Antithetic Variates", "Control Variate"):
            bs_ref = (
                black_scholes_call(S0, K, T, r, sigma)
                if option_type == "call"
                else black_scholes_put(S0, K, T, r, sigma)
            )

        st.markdown("### // RESULT")
        st.markdown("<hr/>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1.4, 1, 1])
        with col_a:
            st.markdown(
                f"""
                <div class="brutal-card" style="background:var(--ink); color:#fff;">
                  <span class="brutal-label" style="color:var(--accent-2);">// ESTIMATED PRICE</span>
                  <div class="brutal-value" style="color:#fff; font-size:64px;">{price:.4f}</div>
                  <div style="margin-top:10px; font-size:11px; letter-spacing:0.2em; color:#bdbdbd;">
                    units of currency  //  discounted expected payoff
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""
                <div class="brutal-card">
                  <span class="brutal-label">// RUNTIME</span>
                  <div class="brutal-value" style="font-size:36px;">{elapsed*1000:.1f} ms</div>
                  <div style="margin-top:8px; font-size:11px; letter-spacing:0.2em;">
                    paths/sec ~ {int(num_sim/elapsed):,}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c:
            if bs_ref is not None:
                diff = price - bs_ref
                st.markdown(
                    f"""
                    <div class="brutal-card accent">
                      <span class="brutal-label">// BLACK-SCHOLES REF</span>
                      <div class="brutal-value" style="font-size:36px;">{bs_ref:.4f}</div>
                      <div style="margin-top:8px; font-size:11px; letter-spacing:0.2em;">
                        DELTA vs MC = {diff:+.4f}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="brutal-card">
                      <span class="brutal-label">// REFERENCE</span>
                      <div style="font-weight:700;">N / A FOR THIS METHOD</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------------
        # GREEKS  ::  DELTA / GAMMA  (european call only)
        # ----------------------------------------------------------
        with st.expander("[ + ]  GREEKS  ::  DELTA  /  GAMMA  vs  SPOT"):
            if option_type == "call":
                S_range = np.linspace(S0 * 0.8, S0 * 1.2, 25)

                plt.rcParams.update(
                    {
                        "font.family": "monospace",
                        "axes.edgecolor": "#0a0a0a",
                        "axes.linewidth": 2.0,
                        "axes.grid": True,
                        "grid.color": "#0a0a0a",
                        "grid.linestyle": ":",
                        "grid.alpha": 0.35,
                        "xtick.color": "#0a0a0a",
                        "ytick.color": "#0a0a0a",
                        "axes.labelcolor": "#0a0a0a",
                        "axes.titlesize": 11,
                        "axes.titleweight": "bold",
                        "figure.facecolor": "#ffffff",
                        "axes.facecolor": "#ffffff",
                    }
                )

                deltas = [
                    monte_carlo_delta(S, K, T, r, sigma, int(num_sim), 1.0, "call")
                    for S in S_range
                ]
                gammas = [
                    monte_carlo_gamma_crn(S, K, T, r, sigma, int(num_sim), 0.5, "call")
                    for S in S_range
                ]

                g1, g2 = st.columns(2)
                with g1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(S_range, deltas, color="#0a0a0a", linewidth=2.5)
                    ax.scatter(S_range, deltas, color="#ff3c00", s=40, zorder=5,
                               edgecolor="#0a0a0a", linewidth=1.5)
                    ax.set_xlabel("SPOT PRICE", fontsize=9, labelpad=8)
                    ax.set_ylabel("DELTA", fontsize=9, labelpad=8)
                    ax.set_title("[ DELTA  vs  SPOT ]", loc="left", pad=12)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                with g2:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.plot(S_range, gammas, color="#0a0a0a", linewidth=2.5)
                    ax2.scatter(S_range, gammas, color="#ffe600", s=40, zorder=5,
                                edgecolor="#0a0a0a", linewidth=1.5)
                    ax2.set_xlabel("SPOT PRICE", fontsize=9, labelpad=8)
                    ax2.set_ylabel("GAMMA", fontsize=9, labelpad=8)
                    ax2.set_title("[ GAMMA  vs  SPOT ]", loc="left", pad=12)
                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="brutal-card">
                      <span class="brutal-label">// NOTICE</span>
                      <div style="font-weight:700;">
                        DELTA  /  GAMMA  PLOTS  AVAILABLE  FOR  EUROPEAN  CALL  ONLY
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="brutal-foot">
      <span>// MONTE CARLO ENGINE  v.01</span>
      <span>STATUS :: OK</span>
      <span>BUILD :: BRUTALIST</span>
      <span>STREAMLIT // NUMPY // SCIPY</span>
    </div>
    """,
    unsafe_allow_html=True,
)
