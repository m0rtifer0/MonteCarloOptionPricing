import numpy as np
from core.monte_carlo_exotic import (
    monte_carlo_asian_arithmetic,
    monte_carlo_asian_geometric,
    asian_geometric_call_analytic,
)

def test_asian_arithmetic_call():
    price = monte_carlo_asian_arithmetic(
        S0=100, K=100, T=1, r=0.05, sigma=0.2,
        num_steps=100, num_simulations=10000, option_type='call'
    )
    assert price > 0

def test_asian_geometric_mc_vs_analytic():
    mc_price = monte_carlo_asian_geometric(
        S0=100, K=100, T=1, r=0.05, sigma=0.2,
        num_steps=100, num_simulations=10000, option_type='call'
    )
    analytic_price = asian_geometric_call_analytic(
        S0=100, K=100, T=1, r=0.05, sigma=0.2
    )
    error = abs(mc_price - analytic_price)
    assert error < 0.2