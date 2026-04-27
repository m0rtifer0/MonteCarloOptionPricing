import numpy as np
from core.black_scholes import black_scholes_call, black_scholes_put

def monte_carlo_antithetic(S0, K, T, r, sigma, num_simulations=10000, option_type='call'):
    """
    Monte Carlo pricing using Antithetic Variates to reduce variance.
    """
    # Make sure the number of simulations is even
    if num_simulations % 2 != 0:
        num_simulations += 1

    half_n = num_simulations // 2
    Z = np.random.standard_normal(half_n)
    Z_antithetic = -Z
    Z_all = np.concatenate([Z, Z_antithetic])

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_all)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return np.mean(discounted)

def monte_carlo_control_variate(S0, K, T, r, sigma, num_simulations=10000, option_type='call'):
    """
    Monte Carlo simulation with control variates using S_T as the control variable.
    Dynamically calculates beta = Cov(Y, X) / Var(X) and applies correction.
    """
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
        bs_value = black_scholes_call(S0, K, T, r, sigma)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
        bs_value = black_scholes_put(S0, K, T, r, sigma)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    Y = discounted
    X = ST

    expected_X = S0 * np.exp(r * T)

    # Numerical stability
    cov_YX = np.cov(Y, X, ddof=1)[0, 1]
    var_X = np.var(X, ddof=1)
    beta = cov_YX / (var_X + 1e-8)  # avoid division by near-zero
    beta = np.clip(beta, -5, 5)     # prevent extreme beta values

    corrected = np.mean(Y) + beta * (expected_X - np.mean(X))

    assert np.isfinite(corrected), "Corrected estimate is NaN or infinite."

    return corrected
