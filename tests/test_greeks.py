import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.greeks import monte_carlo_delta, monte_carlo_gamma, monte_carlo_gamma_crn

def test_monte_carlo_delta_call():
    delta = monte_carlo_delta(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                               num_simulations=50000, h=1.0, option_type='call')
    assert 0.3 < delta < 0.7  # ATM call

def test_monte_carlo_gamma_call():
    gamma = monte_carlo_gamma(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                               num_simulations=50000, h=1.0, option_type='call')
    assert gamma > 0 and gamma < 0.1

def test_monte_carlo_gamma_crn():
    gamma = monte_carlo_gamma_crn(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                   num_simulations=50000, h=1.0, option_type='call')
    assert gamma > 0 and gamma < 0.1
