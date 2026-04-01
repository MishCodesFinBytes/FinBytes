# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is FinBytes

FinBytes is an educational repository with two main parts:
1. **Python modules** (`FinBytes/`) — standalone financial/Python teaching examples (not an installable package)
2. **Jekyll blog** (`docs/`) — GitHub Pages site at https://mishcodesfinbytes.github.io/FinBytes/

License: CC0 1.0 Universal

## Commands

### Jekyll blog (local dev)
```bash
cd docs && bundle exec jekyll serve --future
```

### Running Python modules
Each module is self-contained — run directly:
```bash
python FinBytes/CurrencyDashboard.py
python FinBytes/pers_fin_dashboard/dashboard.py --file data.csv --month 2023-03
```

### Tests
```bash
# pytest-based tests exist in several modules
pytest FinBytes/test_samples/
pytest FinBytes/TestingWithMocks/
pytest FinBytes/pers_fin_dashboard/
```

## Python dependencies (no requirements.txt)

There is no requirements.txt or pyproject.toml. Key libraries used across modules:
- **Data**: pandas, numpy, scipy, statsmodels
- **Visualization**: matplotlib, plotly, dash
- **Finance**: yfinance, ta (technical analysis), prophet
- **ML**: scikit-learn, torch, tensorflow
- **APIs**: requests (CoinMarketCap, CoinGecko)
- **NLP**: nltk
- **Testing**: pytest
- **Other**: redis, click, flask

## Architecture

### Python modules (`FinBytes/`)

Each subdirectory is an independent, self-contained example — not a shared library. There are no cross-module imports. Common pattern: Load data → Process → Visualize → Predict.

Key module groups:
- **Financial analysis**: `InvestmentPortfolioAnalysis/`, `StockAnalysis/`, `RiskMetrics/`, `PredictStockPrices/`, `Timeseries/`
- **Personal finance tools**: `ExpenseTracker/`, `pers_fin_dashboard/`, `RetirementCalc/`, `CreditCardRepaymentCalculator/`, `MiniPersonalBudgetTracker/`
- **Trading/crypto**: `AlgoTrading_MovingAvg_Momentum.py`, `CryptoPortfolioAnalysis_APIs/`, `RealtimeStockPriceTracker.py`
- **Teaching patterns**: `ArgParseCLI/` (CLI patterns), `TestingWithMocks/` (pytest fixtures), `redis_sample/` (Redis demos), `test_samples/` (pytest demos), `ExampleDecorators.py`
- **Data science**: `AnomalyDetection/`, `Clustering/`, `SentimentAnalysis.py`

### Jekyll blog (`docs/`)

- **Config**: `docs/_config.yml` — collections: `cpp`, `quant_lab`, `private`; `future: true`
- **Posts**: `docs/_posts/` (Python/misc/notes), `docs/_cpp/` (C++ series A–J), `docs/_quant_lab/`, `docs/_private/`
- **Layouts**: `docs/_layouts/` — `post.html`, `cpp-post.html` (tabbed), `quant-lab-project.html`
- **Nav**: `docs/_includes/header.html` — dropdowns with CSS embedded as `<style>` block (no separate CSS file). Right-side dropdowns use `nav-submenu-left` class.
- **Index pages**: `python.html`, `cpp.html` (password gated), `misc.html`, `notes.html`, `cheatsheets.html`, `quant-lab.html`, `private.html`

### Liquid date comparisons
All date comparisons must use `{% assign pd = p.date | date: "%Y-%m-%d" %}` then compare as strings — never compare `p.date` directly to a string.

## Git workflow
Branch strategy: always work on `working` branch, merge to `master` after
Commit sequence:
1. git add .
2. git commit -m "<descriptive message>"
3. git push origin working
4. git checkout master
5. git merge working --no-verify -m "<same message>"
6. git push origin master
7. git checkout working

When asked to "commit and deploy", run all 7 steps in order.
Never commit files in docs/_site/
Always stay on working branch when done.
