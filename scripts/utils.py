import pandas as pd
import numpy as np
import math
import pygal

BASELINE_INFLATION_RATE = 0.0434
TUNING_CONSTANT_C = math.pi
TOTAL_CIRCULATING_SOL = 602_997_606

# alpenglow network parameters
SLOTS_PER_EPOCH = 432_000
SECONDS_PER_SLOT = 0.4
EPOCH_DURATION_DAYS = (SLOTS_PER_EPOCH * SECONDS_PER_SLOT) / (24 * 3600)  # ~2 days
SLOTS_PER_YEAR = 365.25 * 24 * 3600 / SECONDS_PER_SLOT  # ~78.8M slots

def load_and_clean_data(path: str) -> pd.DataFrame:
    """load csv data and convert numeric columns to proper types"""
    df = pd.read_csv(path)
    numeric_cols = [
        'Active Stake (SOL)', 'Issuance Revenue (SOL)', 'Jito MEV Revenue (SOL)',
        'Block Rewards Revenue (SOL)', 'Total Revenue (SOL)', 
        'Server Cost (SOL)', 'Voting Cost (SOL)', 'Profit(SOL)'
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Active Stake (SOL)', 'Total Revenue (SOL)'])
    return df

def calc_modified_gini(df: pd.DataFrame) -> float:
    """
    calculate adjusted gini coefficient using Raffinetti et al. reformulation
    that handles negative profits without discarding data
    
    this approach treats negative profits as "negative contributions" that
    increase overall variability, keeping the index bounded in [0,1]
    
    returns:
        float: adjusted gini coefficient (0-1)
    """
    profits = df['Profit(SOL)'].values
    n = len(profits)
    
    # calculate total positive and negative profits
    positive_profits = np.sum(np.maximum(profits, 0))  # T+
    negative_profits = np.abs(np.sum(np.minimum(profits, 0)))  # |T-|
    
    # calculate adjusted mean that includes absolute value of negatives
    adjusted_mean = (positive_profits + negative_profits) / n  # μ*
    
    # vectorized calculation of absolute mean difference: Δ = (1/n²) * Σᵢ Σⱼ |pᵢ - pⱼ|
    abs_diff_matrix = np.abs(profits[:, None] - profits)
    absolute_mean_diff = abs_diff_matrix.mean()
    
    # adjusted gini: G* = Δ / (2μ*)
    if adjusted_mean == 0:
        return 0.0
    
    adjusted_gini = absolute_mean_diff / (2 * adjusted_mean)
    return min(adjusted_gini, 1)

def calc_shannon_entropy(df: pd.DataFrame) -> float:
    """calculate normalized shannon entropy from profit distribution"""
    profits = df['Profit(SOL)'].values
    profits = np.array([max(0, p) for p in profits])  # remove negative profits
    total = profits.sum()
    shares = profits / total
    shares = np.where(shares == 0, 1e-10, shares)
    entropy = -np.sum(shares * np.log2(shares))
    
    # normalize by max entropy
    n = len(profits)
    max_entropy = np.log2(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def apply_simd228(df: pd.DataFrame, stake_ratio: float, target_inflation: float = None) -> pd.DataFrame:
    """apply simd-228 proposal with adjusted stake ratio and inflation"""
    df = df.copy()
    
    current_total_stake = df['Active Stake (SOL)'].sum()
    target_total_stake = TOTAL_CIRCULATING_SOL * stake_ratio
    scaling_factor = target_total_stake / current_total_stake
    df['Active Stake (SOL)'] *= scaling_factor

    inflation = target_inflation if target_inflation else BASELINE_INFLATION_RATE
    
    sqrt_psi = math.sqrt(stake_ratio)
    sqrt_2psi = math.sqrt(2 * stake_ratio)
    multiplier = (1 - sqrt_psi + TUNING_CONSTANT_C * max(1 - sqrt_2psi, 0))
    new_inflation = inflation * multiplier
    df['Issuance Revenue (SOL)'] *= new_inflation / inflation
    return df

def apply_multiplier(df: pd.DataFrame, column: str, multiplier: float) -> pd.DataFrame:
    """apply multiplier to specified column"""
    df = df.copy()
    df[column] *= multiplier
    return df

def apply_less_inflation(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """reduce issuance revenue by multiplier"""
    return apply_multiplier(df, 'Issuance Revenue (SOL)', multiplier)

def change_vote_fees(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """adjust voting costs by multiplier"""
    return apply_multiplier(df, 'Voting Cost (SOL)', multiplier)


def calculate_gini_inflation_ratio(df: pd.DataFrame) -> list:
    """calculate gini coefficient for inflation rates from 4.5% to 0.5% in 0.25% intervals"""
    results = []
    inflation_rates = np.arange(0.5, 4.75, 0.25)[::-1]

    for rate in inflation_rates:
        multiplier = rate / BASELINE_INFLATION_RATE / 100.0
        df_adjusted = apply_less_inflation(df, multiplier)
        df_adjusted = recompute_profits(df_adjusted)
        gini = calc_modified_gini(df_adjusted)
        results.append({'inflation_rate': rate, 'gini_coefficient': gini})

    return results

  
def profit_distribution_chart(df: pd.DataFrame, title: str) -> pygal.Bar:
    """create profit distribution chart by stake percentile buckets"""
    stake = df['Active Stake (SOL)'].astype(float)

    # percentile bins with separate top 1%
    percentile_bins = [0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0]
    bucket_labels = [
        '0-20 %ile',
        '20-40 %ile',
        '40-60 %ile',
        '60-80 %ile',
        '80-99 %ile',
        'top 1 %ile'
    ]
    stake_percentiles = stake.rank(method='min', pct=True)
    df['stake_bucket'] = pd.cut(
        stake_percentiles,
        bins=percentile_bins,
        labels=bucket_labels,
        include_lowest=True,
        right=True
    )

    # average profit per bucket
    avg_profit_by_bucket = (
        df.groupby('stake_bucket')['Profit(SOL)']
        .mean()
        .reindex(bucket_labels)
        .fillna(0)
    )


    chart = pygal.Bar(
        title=title,
        x_title='Stake Size Percentile Bucket',
        y_title='Average Profit (SOL)',
        style=pygal.style.BlueStyle,
        width=700,
        height=400,
        show_legend=False,
        x_label_rotation=15,
        print_values=True,
        print_values_position='top',
        value_formatter=lambda x: f'{x:,.0f}',
        human_readable=True,
    )

    chart.add('average profit', [round(v, 2) for v in avg_profit_by_bucket.values])
    chart.x_labels = bucket_labels

    return chart


def recompute_profits(df: pd.DataFrame) -> pd.DataFrame:
    """recalculate total revenue and profit from component parts"""
    df = df.copy()
    df['Total Revenue (SOL)'] = (
        df['Issuance Revenue (SOL)'] +
        df['Jito MEV Revenue (SOL)'] +
        df['Block Rewards Revenue (SOL)']
    )
    df['Profit(SOL)'] = (
        df['Total Revenue (SOL)'] -
        df['Server Cost (SOL)'] -
        df['Voting Cost (SOL)']
    )
    return df
