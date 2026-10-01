# Stock Market Prediction

Predicts short-term BUY/SELL signals for NSE-listed stocks using an ensemble of machine learning models, served through a Streamlit app.

## Features

- Pulls historical price data via `yfinance` for stocks across 3 market-cap categories (large/mid/small cap)
- 10 engineered technical-analysis features: SMA ratios, RSI, MACD, EMA ratio, ATR, momentum, volume ratio, Bollinger Band width
- Target: whether a stock's 5-day forward return exceeds 1%
- **6 models trained per category** (18 total): Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, and a voting Ensemble
- Streamlit app for live predictions plus an EDA/model-insights page

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-316192?style=flat)
![LightGBM](https://img.shields.io/badge/-LightGBM-02569B?style=flat)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![yfinance](https://img.shields.io/badge/-yfinance-702082?style=flat)

## Project Structure

```
stock market predction.ipynb   # Data collection, feature engineering, model training/saving (per category)
app.py                         # Streamlit app — loads category_models/*.pkl and serves live predictions
test_tickers.py                # Scratch script for sanity-checking ticker symbols resolve via yfinance
```

## ⚠️ Setup required before running

This repo currently ships the code but **not the trained model files** (`category_models/*.pkl`, 18 files) or `plots/` (EDA images) — they're excluded because training needs live network access to Yahoo Finance and takes a while to run. To get a working app:

```bash
pip install -r requirements.txt
jupyter notebook "stock market predction.ipynb"   # run all cells — this generates category_models/ and plots/
streamlit run app.py
```

## Notes

Model-saving naming convention (`category_models/{MODEL}_{category}.pkl`), the feature list/order, and scaled-vs-unscaled handling between tree models (RF/XGB/LGBM, unscaled) and linear models (LR/SVM, scaled) are all consistent between the notebook and `app.py` — verified directly against each other.
