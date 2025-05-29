# tests/test_pricing_accuracy.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from core.black_scholes import black_scholes_call, black_scholes_put
from core.monte_carlo_basic import monte_carlo_option_pricing

def test_black_scholes_call_price():
    price = black_scholes_call(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    assert abs(price - 10.4506) < 0.01

def test_black_scholes_put_price():
    price = black_scholes_put(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    assert abs(price - 5.5735) < 0.01

def test_monte_carlo_call_convergence():
    np.random.seed(42)
    mc_price = monte_carlo_option_pricing(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                           num_simulations=100000, option_type='call')
    bs_price = black_scholes_call(100, 100, 1, 0.05, 0.2)
    assert abs(mc_price - bs_price) < 0.5
