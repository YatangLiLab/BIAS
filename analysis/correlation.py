"""
CSV Column Correlation Analyzer
- Minimal resource usage
- Computes multiple correlation metrics
- Generates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import probplot 
import argparse
import sys

def load_data(file1, file2, col1, col2):
    """
    Efficiently load specified columns from two CSV files
    
    Args:
        file1: Path to first CSV file
        file2: Path to second CSV file  
        col1: Column name in first file
        col2: Column name in second file
    
    Returns:
        x, y: Data arrays for the two columns
    """
    try:
        # Read only required columns with memory-efficient dtype
        df1 = pd.read_csv(file1, usecols=[col1], dtype=np.float32)
        df2 = pd.read_csv(file2, usecols=[col2], dtype=np.float32)
    except Exception as e:
        print(f"Data loading error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Merge and clean data
    data = pd.concat([df1, df2], axis=1).dropna()
    if len(data) == 0:
        print("Error: No valid data after merge", file=sys.stderr)
        sys.exit(1)
        
    return data[col1].values, data[col2].values

def calculate_statistics(x, y):
    """
    Compute comprehensive correlation statistics
    
    Args:
        x, y: Input data arrays
    
    Returns:
        Dictionary containing statistical measures
    """
    stats_ = {
        # Correlation coefficients
        'pearson_r': stats.pearsonr(x, y)[0],
        'pearson_p': stats.pearsonr(x, y)[1],
        'spearman_r': stats.spearmanr(x, y)[0],
        'spearman_p': stats.spearmanr(x, y)[1],
        'kendall_tau': stats.kendalltau(x, y)[0],
        'kendall_p': stats.kendalltau(x, y)[1],
        
        # Additional metrics
        'r_squared': stats.pearsonr(x, y)[0]**2,
        'covariance': np.cov(x, y)[0, 1],
        'x_mean': np.mean(x),
        'y_mean': np.mean(y),
        'x_median': np.median(x),
        'y_median': np.median(y),
        'x_std': np.std(x),
        'y_std': np.std(y),
        'sample_size': len(x),
        'x_range': (np.min(x), np.max(x)),
        'y_range': (np.min(y), np.max(y))
    }
    return stats_

def generate_plots(x, y, x_name, y_name, stats_):
    """
    Create professional correlation visualization
    
    Args:
        x, y: Data arrays
        x_name, y_name: Axis labels
        stats: Precomputed statistics
    
    Returns:
        Path to saved image file
    """
    plt.figure(figsize=(16, 12))
    sns.set_style("whitegrid")
    plt.suptitle(f"Correlation Analysis: {x_name} vs {y_name}\n"
                f"Pearson r = {stats_['pearson_r']:.3f} (p = {stats_['pearson_p']:.2e})", 
                y=1.02)
    
    # Scatter plot with regression line
    ax1 = plt.subplot(2, 2, 1)
    sns.regplot(x=x, y=y, 
               scatter_kws={'s':15, 'alpha':0.6, 'color':'#1f77b4'}, 
               line_kws={'color':'#d62728', 'linewidth':2})
    ax1.set(xlabel=x_name, ylabel=y_name)
    ax1.annotate(f"n = {stats_['sample_size']}", 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round", fc="white"))
    
    # Residual plot
    ax2 = plt.subplot(2, 2, 2)
    sns.residplot(x=x, y=y, lowess=True,
                 scatter_kws={'s':15, 'alpha':0.6},
                 line_kws={'color':'#d62728'})
    ax2.set(xlabel=x_name, ylabel="Residuals")
    
    # Distribution comparison
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(x, color='#1f77b4', kde=True, label=x_name, alpha=0.5)
    sns.histplot(y, color='#ff7f0e', kde=True, label=y_name, alpha=0.5)
    ax3.legend()
    ax3.set(xlabel="Value", ylabel="Density", title="Distribution Comparison")
    
    # QQ plot for normality check
    ax4 = plt.subplot(2, 2, 4)
    probplot(x, dist="norm", plot=ax4)
    ax4.set_title(f"Normality Check (Q-Q Plot) for {x_name}")
    
    plt.tight_layout()
    output_file = f"correlation_{x_name}_vs_{y_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    return output_file

def print_statistics(stats_, x_name, y_name):
    """Display formatted statistical results"""
    print("\n=== BASIC STATISTICS ===")
    print(f"{x_name}:")
    print(f"  Mean = {stats_['x_mean']:.4f} ± {stats_['x_std']:.4f}")
    print(f"  Median = {stats_['x_median']:.4f}")
    print(f"  Range = [{stats_['x_range'][0]:.4f}, {stats_['x_range'][1]:.4f}]")
    
    print(f"\n{y_name}:")
    print(f"  Mean = {stats_['y_mean']:.4f} ± {stats_['y_std']:.4f}")
    print(f"  Median = {stats_['y_median']:.4f}")
    print(f"  Range = [{stats_['y_range'][0]:.4f}, {stats_['y_range'][1]:.4f}]")
    
    print(f"\nSample size = {stats_['sample_size']}")
    
    print("\n=== CORRELATION METRICS ===")
    print(f"Pearson r = {stats_['pearson_r']:.4f} (p = {stats_['pearson_p']:.2e})")
    print(f"Spearman r = {stats_['spearman_r']:.4f} (p = {stats_['spearman_p']:.2e})")
    print(f"Kendall τ = {stats_['kendall_tau']:.4f} (p = {stats_['kendall_p']:.2e})")
    print(f"R² = {stats_['r_squared']:.4f}")
    print(f"Covariance = {stats_['covariance']:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between columns in two CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    parser.add_argument('col1', help='Column name in first file')
    parser.add_argument('col2', help='Column name in second file')
    parser.add_argument('--verbose', '-v', default=True, 
                       help='Show detailed processing information')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Analyzing correlation between {args.col1} (from {args.file1}) "
              f"and {args.col2} (from {args.file2})...")
    
    # 1. Data loading
    x, y = load_data(args.file1, args.file2, args.col1, args.col2)
    
    # 2. Statistical analysis
    stats_ = calculate_statistics(x, y)
    
    # 3. Results display
    print_statistics(stats_, args.col1, args.col2)
    
    # 4. Visualization
    output_file = generate_plots(x, y, args.col1, args.col2, stats_)
    
    if args.verbose:
        print(f"\nVisualization saved to: {output_file}")

if __name__ == "__main__":
    main()