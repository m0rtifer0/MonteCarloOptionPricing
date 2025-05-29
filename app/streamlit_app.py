import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.monte_carlo_basic import monte_carlo_option_pricing
from core.variance_reduction import monte_carlo_antithetic, monte_carlo_control_variate
from core.black_scholes import black_scholes_call, black_scholes_put
from core.monte_carlo_exotic import (
    monte_carlo_asian_arithmetic,
    monte_carlo_asian_geometric,
    asian_geometric_call_analytic
)
from core.greeks import monte_carlo_delta, monte_carlo_gamma_crn

# ----------------------------------------------------------
st.set_page_config(page_title="Option Pricing App", layout="centered")
st.title("Monte Carlo Option Pricing App")

# --- Inputs ---
option_type = st.selectbox("Option Type", ["call", "put"])
method = st.selectbox("Pricing Method", [
    "Basic Monte Carlo",
    "Antithetic Variates",
    "Control Variate",
    "Asian Arithmetic",
    "Asian Geometric (MC)",
    "Asian Geometric (Analytic)"
])

S0 = st.number_input("Initial Stock Price (S₀)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T, in years)", value=1.0)
r = st.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
num_sim = st.number_input("Number of Simulations", min_value=1000, value=100000)

# ----------------------------------------------------------
if st.button("Calculate Price"):

    price = None

    if method == "Basic Monte Carlo":
        price = monte_carlo_option_pricing(S0, K, T, r, sigma, int(num_sim), option_type)

    elif method == "Antithetic Variates":
        price = monte_carlo_antithetic(S0, K, T, r, sigma, int(num_sim), option_type)

    elif method == "Control Variate":
        price = monte_carlo_control_variate(S0, K, T, r, sigma, int(num_sim), option_type)

    elif method == "Asian Arithmetic":
        price = monte_carlo_asian_arithmetic(S0, K, T, r, sigma, int(num_sim), option_type=option_type)

    elif method == "Asian Geometric (MC)":
        price = monte_carlo_asian_geometric(S0, K, T, r, sigma, int(num_sim), option_type=option_type)

    elif method == "Asian Geometric (Analytic)":
        if option_type == "call":
            price = asian_geometric_call_analytic(S0, K, T, r, sigma)
        else:
            st.error("❌ Analytic formula is only available for geometric **call** options.")
            price = None

    if price is not None:
        st.success(f"Estimated Option Price: {price:.4f}")

        # --- Optional: Plot Delta and Gamma ---
        with st.expander("📐 Show Delta / Gamma Plots (European Call Only)"):
            if option_type == "call":
                S_range = np.linspace(S0 * 0.8, S0 * 1.2, 25)

                def plot_delta(S_range, K, T, r, sigma, num_sim, h=1.0):
                    deltas = []
                    for S in S_range:
                        delta = monte_carlo_delta(S, K, T, r, sigma, num_sim, h, option_type="call")
                        deltas.append(delta)
                    fig, ax = plt.subplots()
                    ax.plot(S_range, deltas, marker='o')
                    ax.set_xlabel("Spot Price")
                    ax.set_ylabel("Delta")
                    ax.set_title("Option Delta vs Spot Price")
                    ax.grid(True)
                    return fig

                def plot_gamma(S_range, K, T, r, sigma, num_sim, h=0.5):
                    gammas = []
                    for S in S_range:
                        gamma = monte_carlo_gamma_crn(S, K, T, r, sigma, num_sim, h, option_type="call")
                        gammas.append(gamma)
                    fig, ax = plt.subplots()
                    ax.plot(S_range, gammas, marker='o', color='purple')
                    ax.set_xlabel("Spot Price")
                    ax.set_ylabel("Gamma")
                    ax.set_title("Option Gamma vs Spot Price")
                    ax.grid(True)
                    return fig

                st.write("### Delta vs Spot Price")
                fig_delta = plot_delta(S_range, K, T, r, sigma, int(num_sim), h=1.0)
                st.pyplot(fig_delta)

                st.write("### Gamma vs Spot Price")
                fig_gamma = plot_gamma(S_range, K, T, r, sigma, int(num_sim), h=0.5)
                st.pyplot(fig_gamma)
            else:
                st.warning("Delta/Gamma plots are only available for **European Call** options.")