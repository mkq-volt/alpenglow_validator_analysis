# alpenglow validator analysis

analysis of solana validator economics and distribution patterns.

## analysis

the main analysis is contained in `notebooks/profitability.ipynb`, which examines validator stake distributions, operational costs, and profitability under different scenarios.

## data structure

the dataset in `data/validator_profit.csv` is sourced from [Helius' analysis](https://www.helius.dev/blog/simd-228) contains validator profitability metrics with the following structure:
- validator stake amounts
- operational costs
- revenue calculations
- profit margins

## project structure

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
