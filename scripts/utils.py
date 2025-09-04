import pandas as pd
import numpy as np
import math
import pygal
from IPython.display import SVG, display

from pygal.style import BlueStyle, Style
from scripts.linebar import LineBar

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

    custom_style = Style(
        background=BlueStyle.background,
        plot_background=BlueStyle.plot_background,
        foreground=BlueStyle.foreground,
        foreground_strong=BlueStyle.foreground_strong,
        foreground_subtle=BlueStyle.foreground_subtle,
        opacity=BlueStyle.opacity,
        opacity_hover=BlueStyle.opacity_hover,
        transition=BlueStyle.transition,
        colors=BlueStyle.colors,
        value_font_size=15,      # small value labels
        label_font_size=15,     # small axis labels
        major_label_font_size=15,
        legend_font_size=15,    # small legend
        tooltip_font_size=15,
    )


    chart = pygal.Bar(
        title=title,
        x_title='Stake Size Percentile Bucket',
        y_title='Average Profit (SOL)',
        style=custom_style,
        width=1200,   # much wider
        height=700,   # much taller
        show_legend=True,
        x_label_rotation=15,
        legend_at_bottom=True,
        print_values=True,
        print_values_position='top',
        value_formatter=lambda x: f'{x:,.0f}',
        human_readable=True,
    )

    chart.add('status quo (gini coefficient: 0.9314)', [round(v, 2) for v in avg_profit_by_bucket.values])
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

def get_custom_chart_style(value_font=15):
    """return standardized pygal style configuration"""
    return Style(
        background=BlueStyle.background,
        plot_background=BlueStyle.plot_background,
        foreground=BlueStyle.foreground,
        foreground_strong=BlueStyle.foreground_strong,
        foreground_subtle=BlueStyle.foreground_subtle,
        opacity=BlueStyle.opacity,
        opacity_hover=BlueStyle.opacity_hover,
        transition=BlueStyle.transition,
        colors=BlueStyle.colors,
        value_font_size=value_font,
        label_font_size=15,
        major_label_font_size=15,
        legend_font_size=15,
        tooltip_font_size=15,
    )

def create_stake_bins():
    """return standard stake bins and labels for distribution analysis"""
    stake_bins = [0, 1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, float('inf')]
    bin_labels = ['<1k','1k-10k','10k-50k','50k-100k','100k-250k','250k-500k','500k-1m','>1m']
    return stake_bins, bin_labels

def get_percentile_buckets(df, num_buckets=5):
    """assign stake buckets by percentile, with top 1% as a separate bucket"""
    stake = df['Active Stake (SOL)'].astype(float)
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
    df = df.copy()
    df['stake_bucket'] = pd.cut(
        stake_percentiles,
        bins=percentile_bins,
        labels=bucket_labels,
        include_lowest=True,
        right=True
    )
    return df, bucket_labels

def bucket_profits_with_top1(df, num_buckets=5):
    """create profit buckets with separate top 1% bucket"""
    df = df[['Active Stake (SOL)', 'Profit(SOL)']].dropna().sort_values('Active Stake (SOL)').reset_index(drop=True)
    n = len(df)
    top1_cutoff = int(np.ceil(n * 0.99))
    
    bucket_labels = [f'{int(i*100/num_buckets)}-{int((i+1)*100/num_buckets)}%ile' for i in range(num_buckets - 1)]
    bucket_labels.append('80-99%ile')
    bucket_labels.append('top 1%')
    
    bucket_assignment = pd.Series([None] * n)
    
    if top1_cutoff > 0:
        quantile_labels = [f'{int(i*100/num_buckets)}-{int((i+1)*100/num_buckets)}%ile' for i in range(num_buckets - 1)]
        quantile_labels.append('80-99%ile')
        bucket_assignment.iloc[:top1_cutoff] = pd.qcut(
            np.arange(top1_cutoff), num_buckets, labels=quantile_labels
        )
    
    bucket_assignment.iloc[top1_cutoff:] = 'top 1%'
    df['bucket'] = bucket_assignment

    grouped = df.groupby('bucket', observed=True)['Profit(SOL)']
    avg_profits = grouped.mean().reindex(bucket_labels).tolist()
    total_profits = grouped.sum().reindex(bucket_labels).tolist()
    
    return bucket_labels, avg_profits, total_profits

def create_stake_distribution_chart(df, title="Validators by Stake Bucket"):
    """create stake distribution chart showing validator counts and total stake"""
    stake_bins, bin_labels = create_stake_bins()
    
    stake_binned = pd.cut(
        df['Active Stake (SOL)'],
        bins=stake_bins, labels=bin_labels, right=False, include_lowest=True
    )

    validator_counts = stake_binned.value_counts().reindex(bin_labels, fill_value=0)
    stake_per_bucket = (
        df.groupby(stake_binned, observed=False)['Active Stake (SOL)']
        .sum()
        .reindex(bin_labels, fill_value=0)
    )

    counts = [int(validator_counts.get(lb, 0)) for lb in bin_labels]
    stake_sol = [float(stake_per_bucket.get(lb, 0.0)) for lb in bin_labels]

    config = pygal.Config()
    config.human_readable = True
    config.legend_at_bottom = True
    config.legend_at_bottom_columns = 2
    config.x_label_rotation = 30
    config.print_values = False
    config.interpolate = False
    config.value_formatter = lambda x: str(int(x))
    
    def fmt_sol(value):
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}k"
        return str(int(value))
    config.secondary_value_formatter = fmt_sol

    primary_range = (0, int(max(5, (max(counts) if counts else 0) * 1.1)))
    secondary_range = (0, max(1.0, (max(stake_sol) if stake_sol else 0.0) * 1.1))

    chart = LineBar(
        config,
        width=900,
        height=420,
        title=title,
        x_title="Active Stake Range (SOL)",
        y_title="Number of Validators",
        y_title_secondary="Total Stake (SOL)",
        legend_box_size=10,
        range=primary_range,
        secondary_range=secondary_range,
        style=pygal.style.DefaultStyle(value_font_size=10),
        human_readable=True,
        format_secondary_value=fmt_sol,
    )

    chart.x_labels = bin_labels + [""]  # prevent last bar overlap
    chart.add("number of validators", counts, plotas='bar')
    chart.add("total stake (SOL)", stake_sol, plotas='line', secondary=True)

    return chart

def create_revenue_breakdown_chart(df, top_n=100, title_prefix="Top", chart_title=None):
    """create stacked bar chart showing revenue breakdown by source"""
    if chart_title is None:
        chart_title = f'{title_prefix} {top_n} Validator Revenue Breakdown'
    
    if title_prefix.lower() == "top":
        df_sorted = df.sort_values('Total Revenue (SOL)', ascending=False).head(top_n)
    else:  # bottom
        df_bottom = df.sort_values('Total Revenue (SOL)', ascending=True).head(top_n)
        df_sorted = df_bottom.sort_values('Total Revenue (SOL)', ascending=False)

    block_rewards = df_sorted['Total Revenue (SOL)'] * df_sorted['Block Rewards Revenue as % of Total Revenue (%)']
    mev_revenue = df_sorted['Jito MEV Revenue (SOL)'].astype(float)
    issuance = df_sorted['Issuance Revenue (SOL)'].astype(float)

    chart = pygal.StackedBar(
        title=chart_title,
        x_title='Validator',
        y_title='Revenue (SOL)',
        style=get_custom_chart_style(),
        width=1200,
        height=700,
        x_label_rotation=30,
        show_legend=True,
        human_readable=True,
        legend_at_bottom=True,
        print_values=False,
    )

    chart.add('fees', [round(v, 2) for v in block_rewards])
    chart.add('mev revenue', [round(v, 2) for v in mev_revenue])
    if issuance.sum() > 0:
        chart.add('issuance', [round(v, 2) for v in issuance])

    return chart

def create_gini_inflation_chart(df, title='Gini Coefficient vs Inflation Rate'):
    """create line chart showing gini coefficient vs inflation rate"""
    gini_improvements = calculate_gini_inflation_ratio(df)
    
    chart = pygal.Line(
        title=title,
        x_title='Inflation Rate (%)',
        y_title='Gini Coefficient',
        style=get_custom_chart_style(),
        width=1200,
        height=700,
        show_legend=False,
        x_label_rotation=30,
        human_readable=True,
        interpolate='cubic'
    )

    inflation_rates = [float(imp['inflation_rate']) for imp in gini_improvements]
    gini_coefficients = [float(imp['gini_coefficient']) for imp in gini_improvements]

    chart.x_labels = [f"{rate:.2f}" for rate in inflation_rates]
    chart.add('gini coefficient', gini_coefficients)

    return chart

def create_profit_comparison_chart(scenarios, scenario_labels, title='Average Validator Profit by Stake Size'):
    """create multi-bar chart comparing profit scenarios by stake percentiles"""
    def get_bucket_means(df):
        stake = df['Active Stake (SOL)'].astype(float)
        percentile_bins = [0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0]
        bucket_labels = ['0-20 %ile', '20-40 %ile', '40-60 %ile', '60-80 %ile', '80-99 %ile', 'top 1 %ile']
        
        stake_percentiles = stake.rank(method='min', pct=True)
        df = df.copy()
        df['stake_bucket'] = pd.cut(
            stake_percentiles,
            bins=percentile_bins,
            labels=bucket_labels,
            include_lowest=True,
            right=True
        )
        means = (
            df.groupby('stake_bucket', observed=True)['Profit(SOL)']
            .mean()
            .reindex(bucket_labels)
            .fillna(0)
        )
        return means, bucket_labels

    scenario_means = []
    bucket_labels = None
    for df in scenarios:
        means, bucket_labels = get_bucket_means(df)
        scenario_means.append(means)

    chart = pygal.Bar(
        title=title,
        x_title='Stake Size Percentile Bucket',
        y_title='Average Profit (SOL)',
        style=get_custom_chart_style(10),
        width=1200,
        height=700,
        show_legend=True,
        x_label_rotation=15,
        legend_at_bottom=True,
        print_values=True,
        print_values_position='top',
        value_formatter=lambda x: f'{x:,.0f}',
        human_readable=True,
    )

    chart.x_labels = bucket_labels
    for i, (means, label) in enumerate(zip(scenario_means, scenario_labels)):
        chart.add(label, [round(v, 2) for v in means.values])

    return chart

def create_simd228_chart(scenarios, scenario_labels, title='SIMD228 Analysis: Average Validator Profit by Stake Size'):
    """create chart for SIMD228 scenario analysis with total profit labels"""
    scenario_data = []
    for df in scenarios:
        labels, avg_profits, total_profits = bucket_profits_with_top1(df, num_buckets=5)
        scenario_data.append((labels, avg_profits, total_profits))

    chart = pygal.Bar(
        title=title,
        x_title='Stake Size Percentile Bucket',
        y_title='Average Profit (SOL)',
        width=1200,
        height=700,
        show_legend=True,
        legend_at_bottom=True,
        style=get_custom_chart_style(),
        human_readable=True,
        value_formatter=lambda x: f'{x:.0f}',
        show_y_guides=True,
        print_values=True,
        print_values_position='top',
        x_label_rotation=30,
    )

    chart.x_labels = scenario_data[0][0]  # use labels from first scenario
    
    for i, ((labels, avg_profits, total_profits), scenario_label) in enumerate(zip(scenario_data, scenario_labels)):
        chart.add(scenario_label, [
            {'value': avg if not pd.isna(avg) else 0, 'label': f'{int(tot):,}' if not pd.isna(tot) else '0'} 
            for avg, tot in zip(avg_profits, total_profits)
        ])

    return chart
