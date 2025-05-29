# Monte Carlo Option Pricing

This repository provides a comprehensive and modular framework for pricing European and exotic options using Monte Carlo simulations. The implementation includes variance reduction techniques, Greek estimations, and a Streamlit-based UI for interactive exploration.

## Features

- European option pricing using Black-Scholes and Monte Carlo
- Asian options (arithmetic & geometric average) via Monte Carlo and analytic formulas
- Variance reduction: Antithetic variates, Control variates
- Delta and Gamma estimation using finite difference and CRN
- Streamlit interface for user interaction
- Extensive Jupyter Notebooks for convergence, sensitivity, and visualization
- Test suite with pytest
- Utility modules for reproducibility, simulation, and plotting



## Directory Structure

MonteCarloOptionPricing/
│
├── app/
│ └── streamlit_app.py # Streamlit-based user interface
│
├── assets/ # Output plots for documentation or app
│ ├── arithmetic_vs_geometric.png
│ ├── convergence_plot.png
│ ├── delta_vs_spot.png
│ └── gamma_vs_spot.png
│
├── core/ # Core pricing models and financial logic
│ ├── black_scholes.py
│ ├── greeks.py
│ ├── monte_carlo_basic.py
│ ├── monte_carlo_exotic.py
│ ├── payoff.py
│ ├── variance_reduction.py
│ └── volatility_models.py # Reserved for future extensions
│
├── notebooks/ # Jupyter Notebooks for analysis and experiments
│ ├── 1_convergence.ipynb
│ ├── 2_antithetic_vs_basic.ipynb
│ ├── 3_control_variates_test.ipynb
│ ├── 4_exotic_pricing.ipynb
│ ├── 5_volatility_models.ipynb
│ ├── 6_sensitivity_analysis.ipynb
│ ├── 7_visual_analysis.ipynb
│ ├── 8_arithmetic_vs_geometric.ipynb
│ ├── 9_greeks.ipynb
│ └── 10_simple_monte_carlo.ipynb
│
├── tests/ # Unit tests for all modules
│ ├── test_greeks.py
│ ├── test_monte_carlo_exotic.py
│ ├── test_pricing_accuracy.py
│ └── test_variance_reduction.py
│
├── utils/ # Supporting utility functions
│ ├── seed_utils.py
│ ├── simulation_helpers.py
│ └── visualization.py
│
├── .gitignore # Ignore .ipynb_checkpoints, pycache, etc.
├── LICENSE # MIT License
├── README.md # This file
└── requirements.txt # Python dependencies

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/m0rtifer0/MonteCarloOptionPricing
   cd MonteCarloOptionPricing

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
