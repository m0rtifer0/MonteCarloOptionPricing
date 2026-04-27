# MONTE CARLO OPTION PRICING

A modular framework for pricing European and exotic options via Monte Carlo
simulation, with variance reduction techniques, Greek estimators, a Streamlit
terminal in modern brutalist style, and a Jupyter notebook research suite.

---

## FEATURES

- European option pricing via Black-Scholes and Monte Carlo
- Asian options (arithmetic and geometric average) via MC and analytic formulas
- Variance reduction: antithetic variates, control variates
- Delta and Gamma estimation via finite differences with Common Random Numbers
- Streamlit interface (brutalist UI) for interactive exploration
- Jupyter notebooks for convergence, sensitivity, and visual analysis
- Test suite (pytest)
- Utilities for reproducibility, simulation paths, and plotting

---

## INSTALLATION

```bash
git clone https://github.com/m0rtifer0/MonteCarloOptionPricing
cd MonteCarloOptionPricing
pip install -r requirements.txt
```

## RUN

Streamlit terminal:

```bash
streamlit run app/streamlit_app.py
```

Tests:

```bash
pytest tests/
```

---

## STRUCTURE

```
core/        pricing engines, Greeks, payoffs, variance reduction
app/         streamlit UI (brutalist)
notebooks/   research notebooks
tests/       pytest suite
utils/       helpers (seeds, paths, plotting)
assets/      generated figures
```

---

## LICENSE

Released under the MIT License. See [LICENSE](LICENSE).
