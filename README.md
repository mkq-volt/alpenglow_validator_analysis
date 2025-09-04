# Alpenglow Validator Analysis

Analysis of Solana's validator set economics and stake distribution patterns.

## Analysis

The main analysis is contained in `notebooks/profitability.ipynb`, which examines validator stake distributions, operational costs, and profitability under different scenarios.

## Data Structure

The dataset in `data/validator_profit.csv` is sourced from [Helius' analysis](https://www.helius.dev/blog/simd-228) and contains validator profitability metrics with the following structure:
- validator stake amounts
- operational costs
- revenue calculations
- profit margins

## Project Structure

```
alpenglow_validator_analysis/
├── data/
│   └── validator_profit.csv
├── notebooks/
│   └── profitability.ipynb
├── scripts/
│   ├── linebar.py
│   └── utils.py
├── charts/
└── requirements.txt
```

## requirements

python 3.10+ with dependencies listed in `requirements.txt`.
