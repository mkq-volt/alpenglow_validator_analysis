# Solana Validator Distribution Analysis

## 🧠 Project Objective

Analyze and visualize validator-related distributions within the Solana blockchain ecosystem using statistical and graphical methods. Build 4 key probability distributions and compute their Gini coefficients to understand wealth concentration and validator economics.

## 📊 Distributions

1. **Stake Distribution**: Histogram and PDF of validator stake sizes.
2. **Validator Cost Distribution**: Histogram of operational costs (assume $/SOL ratio).
3. **Validator Income Distribution**: Histogram of validator incomes (assume yield per SOL).
4. **Validator Profitability Distribution**: Income minus cost per validator.

## 📈 Gini Coefficient

- Compute for each distribution.
- Visualize with Lorenz Curves.

## 🗂️ Dataset

- `data/validator_stats.csv` (columns: Stake Range, Total Staked, Number of Validators, Median Stake)

## 🛠️ Requirements

- Python 3.10+
- See `requirements.txt`

## 📁 Structure

```
solana-validator-analysis/
│
├── data/
│   └── validator_stats.csv
│
├── notebooks/
│   └── 01_distributions.ipynb
│   └── 02_gini_analysis.ipynb
│
├── scripts/
│   └── generate_distributions.py
│   └── compute_gini.py
│
├── README.md
└── requirements.txt
```

## 🚀 Usage

- Run notebooks in `notebooks/` for interactive analysis.
- Use scripts in `scripts/` for CLI-based analysis.
