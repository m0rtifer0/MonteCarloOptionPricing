# utils/simulation_helpers.py

import numpy as np

def simulate_terminal_price(S0, T, r, sigma, Z):
    """
    Simulates the terminal asset price using geometric Brownian motion.

    Args:
        S0 (float): Initial asset price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        Z (np.ndarray): Standard normal random variables.

    Returns:
        np.ndarray: Simulated terminal prices.
    """
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def generate_paths(S0, T, r, sigma, steps, n_paths):
    """
    Simulates full asset price paths (useful for Asian and Barrier options).

    Args:
        S0 (float): Initial asset price.
        T (float): Time to maturity (in years).
        r (float): Risk-free rate.
        sigma (float): Volatility.
        steps (int): Number of time steps.
        n_paths (int): Number of simulation paths.

    Returns:
        np.ndarray: Simulated price paths (n_paths x (steps + 1)).
    """
    dt = T / steps
    Z = np.random.standard_normal((n_paths, steps))
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    return paths