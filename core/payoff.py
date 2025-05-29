# core/payoff.py

def call_payoff(S, K):
    """ European Call Option Payoff """
    return max(S - K, 0)

def put_payoff(S, K):
    """ European Put Option Payoff """
    return max(K - S, 0)