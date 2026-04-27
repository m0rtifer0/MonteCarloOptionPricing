import numpy as np
from scipy.stats import norm

def monte_carlo_asian_arithmetic(S0, K, T, r, sigma, num_simulations=10000, num_steps=100, option_type='call'):
    dt = T / num_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.standard_normal((num_simulations, num_steps))
    log_paths = np.cumsum(drift + diffusion * Z, axis=1)
    S_paths = S0 * np.exp(log_paths)

    arith_avg = np.mean(S_paths, axis=1)

    if option_type == 'call':
        payoffs = np.maximum(arith_avg - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - arith_avg, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return np.mean(discounted)

def monte_carlo_asian_geometric(S0, K, T, r, sigma, num_simulations=10000, num_steps=100, option_type='call'):
    dt = T / num_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Simulate paths
    Z = np.random.standard_normal((num_simulations, num_steps))
    log_paths = np.cumsum(drift + diffusion * Z, axis=1)
    S_paths = S0 * np.exp(log_paths)

    # Calculate geometric average
    log_avg = np.mean(np.log(S_paths), axis=1)
    geo_avg = np.exp(log_avg)

    if option_type == 'call':
        payoffs = np.maximum(geo_avg - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - geo_avg, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return np.mean(discounted)


def asian_geometric_call_analytic(S0, K, T, r, sigma):
    """
    Analytic price of geometric average Asian call option (European).
    """
    sigma_eff = sigma / np.sqrt(3)
    r_eff = 0.5 * (r - sigma**2 / 6)

    d1 = (np.log(S0 / K) + (r_eff + 0.5 * sigma_eff ** 2) * T) / (sigma_eff * np.sqrt(T))
    d2 = d1 - sigma_eff * np.sqrt(T)

    price = np.exp(-r * T) * (S0 * np.exp(r_eff * T) * norm.cdf(d1) - K * norm.cdf(d2))
    return price


def asian_geometric_put_analytic(S0, K, T, r, sigma):
    """
    Analytic price of geometric average Asian put option (European).
    """
    sigma_eff = sigma / np.sqrt(3)
    r_eff = 0.5 * (r - sigma**2 / 6)

    d1 = (np.log(S0 / K) + (r_eff + 0.5 * sigma_eff ** 2) * T) / (sigma_eff * np.sqrt(T))
    d2 = d1 - sigma_eff * np.sqrt(T)

    price = np.exp(-r * T) * (K * norm.cdf(-d2) - S0 * np.exp(r_eff * T) * norm.cdf(-d1))
    return price
