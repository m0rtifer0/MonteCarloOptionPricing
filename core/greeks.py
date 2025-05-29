from core.monte_carlo_basic import monte_carlo_option_pricing

def monte_carlo_delta(S0, K, T, r, sigma, num_simulations=10000, h=1.0, option_type='call'):
    """Compute Delta using central finite difference method."""
    V_plus = monte_carlo_option_pricing(S0 + h, K, T, r, sigma, num_simulations, option_type)
    V_minus = monte_carlo_option_pricing(S0 - h, K, T, r, sigma, num_simulations, option_type)
    delta = (V_plus - V_minus) / (2 * h)
    return delta

from core.monte_carlo_basic import monte_carlo_option_pricing

def monte_carlo_gamma(S0, K, T, r, sigma, num_simulations=10000, h=1.0, option_type='call'):
    """
    Compute Gamma using central finite difference:
    Γ ≈ (V(S+h) - 2V(S) + V(S−h)) / h²
    """
    V_plus = monte_carlo_option_pricing(S0 + h, K, T, r, sigma, num_simulations, option_type)
    V = monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, option_type)
    V_minus = monte_carlo_option_pricing(S0 - h, K, T, r, sigma, num_simulations, option_type)
    
    gamma = (V_plus - 2 * V + V_minus) / (h ** 2)
    return gamma

import numpy as np

from core.monte_carlo_basic import monte_carlo_option_price_given_Z

def monte_carlo_gamma_crn(S0, K, T, r, sigma, num_simulations=100000, h=1.0, option_type='call'):
    """
    Gamma computation with Common Random Numbers (CRN) to reduce variance.
    Requires a modified pricing function that accepts Z.
    """

    # Common random numbers
    Z = np.random.standard_normal(num_simulations)

    # Compute 3 price estimates with same Z
    V_plus = monte_carlo_option_price_given_Z(S0 + h, K, T, r, sigma, Z, option_type)
    V = monte_carlo_option_price_given_Z(S0, K, T, r, sigma, Z, option_type)
    V_minus = monte_carlo_option_price_given_Z(S0 - h, K, T, r, sigma, Z, option_type)

    gamma = (V_plus - 2 * V + V_minus) / (h ** 2)
    return gamma