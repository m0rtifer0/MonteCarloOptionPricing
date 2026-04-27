# core/monte_carlo_basic.py

import numpy as np
from core.payoff import call_payoff, put_payoff

def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations=10000, option_type='call'):
    """
    Monte Carlo simulation for European option pricing

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility
    - num_simulations: Number of simulations
    - option_type: 'call' or 'put'

    Returns:
    - Estimated option price
    """
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoff)

def monte_carlo_option_price_given_Z(S0, K, T, r, sigma, Z, option_type='call'):
    """
    Pricing using externally provided random Z values (for CRN).
    """
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return np.mean(discounted)
