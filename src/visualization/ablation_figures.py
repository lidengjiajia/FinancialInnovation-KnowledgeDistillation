"""Ablation figure generator for the manuscript (Nature-style).

This module generates the four ablation figures referenced by the paper:
- ablation_temperature.png
- ablation_alpha.png
- ablation_shap_weight.png
- ablation_max_depth.png

It is intentionally file-path agnostic and is meant to be called from `main.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AblationPlotSpec:
    ablation_type: str
    xlabel: str
    filename: str


DEFAULT_SPECS: tuple[AblationPlotSpec, ...] = (
    AblationPlotSpec('temperature', 'Temperature (τ)', 'ablation_temperature.png'),
    AblationPlotSpec('alpha', 'Hard-label weight (α)', 'ablation_alpha.png'),
    AblationPlotSpec('shap_weight', 'SHAP weight (γ)', 'ablation_shap_weight.png'),
    AblationPlotSpec('max_depth', 'Tree depth (d)', 'ablation_max_depth.png'),
)


DATASET_LABELS = {
    'german': 'German',
    'australian': 'Australian',
    'uci': 'UCI',
    'xinwang': 'Xinwang',
}

# Academic-style harmonious palette (colorblind-friendly)
COLORS = {
    'german': '#0072B2',      # Strong blue
    'australian': '#D55E00',  # Vermillion
    'uci': '#009E73',         # Bluish green
    'xinwang': '#CC79A7',     # Reddish purple
}

MARKERS = {
    'german': 'o',
    'australian': 's',
    'uci': '^',
    'xinwang': 'D',
}


def _load_ablation(results_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(results_dir, f'{dataset}_ablation.xlsx')
    if not os.path.exists(path):
        return None
    return pd.read_excel(path)


def _extract(df: pd.DataFrame, ablation_type: str) -> pd.DataFrame:
    return df[df['ablation_type'] == ablation_type].copy()


def generate_ablation_figures(
    *,
    results_dir: str = 'results',
    output_dir: str = os.path.join('results', 'figures'),
    datasets: Iterable[str] = ('german', 'australian', 'uci', 'xinwang'),
    specs: Iterable[AblationPlotSpec] = DEFAULT_SPECS,
) -> list[str]:
    """Generate the four ablation figures used in the manuscript.

    Returns a list of saved PNG paths.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Nature-style base settings (keep deterministic and consistent sizing)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []

    for spec in specs:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))  # Better aspect ratio
        has_data = False

        for dataset in datasets:
            df = _load_ablation(results_dir, dataset)
            if df is None:
                continue
            subset = _extract(df, spec.ablation_type)
            if subset.empty:
                continue

            has_data = True
            x_values = subset['value'].to_numpy()
            y_values = subset['auc_mean'].to_numpy()
            y_std = subset['auc_std'].to_numpy() if 'auc_std' in subset.columns else np.zeros_like(y_values)

            ax.errorbar(
                x_values,
                y_values,
                yerr=y_std,
                label=DATASET_LABELS.get(dataset, str(dataset)),
                color=COLORS.get(dataset, '#333333'),
                marker=MARKERS.get(dataset, 'o'),
                markersize=5,
                linewidth=1.2,
                capsize=2,
                capthick=0.8,
                markeredgewidth=0.8,
                markeredgecolor='white',
            )

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel('AUC')

        # Legend above, horizontal, with up to 4 datasets
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.18),
            ncol=4,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.3,
            columnspacing=0.8,
        )

        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - margin, y_max + margin)

        fig.tight_layout()

        out_path = os.path.join(output_dir, spec.filename)
        fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none', format='png')
        plt.close(fig)
        saved.append(out_path)

    return saved


def generate_rule_effectiveness_figure(
    *,
    rules_dir: str = os.path.join('results', 'rules'),
    output_dir: str = os.path.join('results', 'figures'),
    datasets: Iterable[str] = ('german', 'australian', 'uci', 'xinwang'),
) -> str | None:
    """Generate the rule effectiveness bar chart for the manuscript.
    
    Creates a figure with two subplots:
    (a) Average and maximum rule confidence with 50% baseline
    (b) Proportion of high-confidence rules (>60%)
    
    Returns the saved PNG path, or None if data is missing.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Nature-style settings
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect rule statistics from Excel files
    stats = []
    for ds in datasets:
        xlsx_path = os.path.join(rules_dir, f'{ds}_CBKD_rules.xlsx')
        if not os.path.exists(xlsx_path):
            continue
        df = pd.read_excel(xlsx_path)
        if 'Confidence' not in df.columns:
            continue
        
        # Parse confidence (handle percentage strings like "85.2%")
        conf_col = df['Confidence']
        if conf_col.dtype == object:
            conf_vals = conf_col.str.rstrip('%').astype(float) / 100
        else:
            conf_vals = conf_col
        
        n_rules = len(conf_vals)
        avg_conf = conf_vals.mean()
        max_conf = conf_vals.max()
        n_high = (conf_vals > 0.6).sum()
        high_ratio = n_high / n_rules if n_rules > 0 else 0
        
        stats.append({
            'dataset': ds,
            'label': DATASET_LABELS.get(ds, ds),
            'n_rules': n_rules,
            'avg_conf': avg_conf,
            'max_conf': max_conf,
            'n_high': n_high,
            'high_ratio': high_ratio,
        })
    
    if not stats:
        return None
    
    # Create figure with two subplots - wider for better proportion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    
    labels = [s['label'] for s in stats]
    x = np.arange(len(labels))
    width = 0.35
    
    # Subplot (a): Average and max confidence
    avg_confs = [s['avg_conf'] * 100 for s in stats]
    max_confs = [s['max_conf'] * 100 for s in stats]
    
    bars1 = ax1.bar(x - width/2, avg_confs, width, label='Average', 
                     color='#0072B2', edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, max_confs, width, label='Maximum',
                     color='#D55E00', edgecolor='white', linewidth=0.5)
    
    # Add 50% baseline
    ax1.axhline(y=50, color='#888888', linestyle='--', linewidth=1, label='Random (50%)')
    
    ax1.set_ylabel('Confidence (%)')
    ax1.set_xlabel('')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 110)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    
    # Subplot (b): High-confidence rule ratio
    high_ratios = [s['high_ratio'] * 100 for s in stats]
    colors_list = [COLORS.get(s['dataset'], '#333333') for s in stats]
    
    bars3 = ax2.bar(x, high_ratios, width*1.5, color=colors_list, edgecolor='white', linewidth=0.5)
    
    ax2.set_ylabel('High-Confidence Rules (%)')
    ax2.set_xlabel('')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 110)
    
    # Add value labels with rule counts
    for bar, s in zip(bars3, stats):
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%\n({s["n_high"]}/{s["n_rules"]})', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    
    fig.tight_layout()
    
    out_path = os.path.join(output_dir, 'rule_effectiveness.png')
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none', format='png')
    plt.close(fig)
    
    return out_path


def generate_class_balance_figure(
    *,
    results_dir: str = 'results',
    output_dir: str = os.path.join('results', 'figures'),
    datasets: Iterable[str] = ('german', 'australian', 'uci', 'xinwang'),
) -> str | None:
    """Generate the class balance ablation bar chart for the manuscript.
    
    Creates a grouped bar chart comparing AUC with and without class-balanced weighting.
    The chart clearly shows that CB-KD (with class balance) outperforms SoftLabelKD
    (without class balance) across datasets with varying imbalance levels.
    
    Returns the saved PNG path, or None if data is missing.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Nature-style settings
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect class balance ablation data
    stats = []
    for ds in datasets:
        xlsx_path = os.path.join(results_dir, f'{ds}_ablation.xlsx')
        if not os.path.exists(xlsx_path):
            continue
        df = pd.read_excel(xlsx_path)
        cb_rows = df[df['ablation_type'] == 'class_balance']
        if cb_rows.empty:
            continue
        
        no_cb_row = cb_rows[cb_rows['value'] == 0]
        with_cb_row = cb_rows[cb_rows['value'] == 1]
        
        if no_cb_row.empty or with_cb_row.empty:
            continue
        
        no_cb_auc = no_cb_row['auc_mean'].values[0]
        with_cb_auc = with_cb_row['auc_mean'].values[0]
        
        stats.append({
            'dataset': ds,
            'label': DATASET_LABELS.get(ds, ds),
            'no_cb_auc': no_cb_auc,
            'with_cb_auc': with_cb_auc,
            'improvement': with_cb_auc - no_cb_auc,
        })
    
    if not stats:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    labels = [s['label'] for s in stats]
    x = np.arange(len(labels))
    width = 0.35
    
    # AUC values
    no_cb_aucs = [s['no_cb_auc'] for s in stats]
    with_cb_aucs = [s['with_cb_auc'] for s in stats]
    
    # Create bars - SoftLabelKD on left (lighter), CB-KD on right (darker)
    bars1 = ax.bar(x - width/2, no_cb_aucs, width, label='SoftLabelKD (w/o CB)', 
                   color='#9ECAE1', edgecolor='#4292C6', linewidth=1)
    bars2 = ax.bar(x + width/2, with_cb_aucs, width, label='CB-KD (with CB)',
                   color='#2171B5', edgecolor='#084594', linewidth=1)
    
    ax.set_ylabel('AUC')
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Set y-axis to start from a reasonable value to show differences clearly
    all_values = no_cb_aucs + with_cb_aucs
    y_min = min(all_values) - 0.03
    y_max = max(all_values) + 0.03
    ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    
    for bar, s in zip(bars2, stats):
        height = bar.get_height()
        improvement = s['improvement']
        sign = '+' if improvement >= 0 else ''
        ax.annotate(f'{height:.3f}\n({sign}{improvement:.3f})', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    
    fig.tight_layout()
    
    out_path = os.path.join(output_dir, 'ablation_class_balance.png')
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none', format='png')
    plt.close(fig)
    
    return out_path