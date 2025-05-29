import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.variance_reduction import monte_carlo_antithetic, monte_carlo_control_variate
from core.black_scholes import black_scholes_call

def test_antithetic_mc():
    price = monte_carlo_antithetic(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                    num_simulations=10000, option_type='call')
    assert price > 0

def test_control_variate_improvement():
    bs_price = black_scholes_call(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    cv_price = monte_carlo_control_variate(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                           num_simulations=10000, option_type='call')
    assert abs(cv_price - bs_price) < 0.5