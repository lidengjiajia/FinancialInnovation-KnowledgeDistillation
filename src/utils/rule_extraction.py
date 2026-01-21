#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rule extraction utilities.

This module extracts IF--THEN rules from trained Decision Tree students.

Note: To keep the workspace tidy, this module intentionally does not expose its
own command-line entrypoint. Use `main.py rules ...` instead.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def extract_rules_from_tree(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str] = ['Non-default', 'Default']
) -> List[Dict]:
    """
    Extract IF-THEN rules from a trained Decision Tree.
    
    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
    
    Returns:
        List of rule dictionaries
    """
    tree_ = tree.tree_
    rules = []
    
    def traverse(node_id, conditions):
        """Recursively traverse tree to extract rules."""
        if tree_.feature[node_id] == -2:  # Leaf node
            values = tree_.value[node_id][0]
            total_samples = sum(values)
            predicted_class = np.argmax(values)
            confidence = values[predicted_class] / total_samples if total_samples > 0 else 0
            
            rule = {
                'rule_id': f"R{len(rules) + 1}",
                'conditions': conditions.copy(),
                'prediction': class_names[predicted_class],
                'prediction_code': int(predicted_class),
                'samples': int(total_samples),
                'confidence': float(confidence),
                'class_distribution': {
                    class_names[i]: int(v) for i, v in enumerate(values)
                }
            }
            rules.append(rule)
        else:
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_name = feature_names[feature_idx]
            
            # Left child (<=)
            left_conditions = conditions + [(feature_name, '<=', threshold)]
            traverse(tree_.children_left[node_id], left_conditions)
            
            # Right child (>)
            right_conditions = conditions + [(feature_name, '>', threshold)]
            traverse(tree_.children_right[node_id], right_conditions)
    
    traverse(0, [])
    return rules


def rules_to_text(rules: List[Dict], max_rules: Optional[int] = None) -> str:
    """
    Convert rules to human-readable text format.
    
    Args:
        rules: List of rule dictionaries
        max_rules: Maximum number of rules to include
    
    Returns:
        Formatted string with IF-THEN rules
    """
    if max_rules:
        rules = sorted(rules, key=lambda r: (-r['confidence'], -r['samples']))[:max_rules]
    
    lines = ["=" * 70]
    lines.append("DECISION RULES EXTRACTED FROM SHAP-KD MODEL")
    lines.append("=" * 70)
    lines.append(f"Total rules: {len(rules)}")
    lines.append("")
    
    for rule in rules:
        lines.append(f"[{rule['rule_id']}] (Samples: {rule['samples']}, "
                    f"Confidence: {rule['confidence']:.2%})")
        
        if rule['conditions']:
            conditions_str = " AND\n       ".join(
                [f"{feat} {op} {thresh:.4f}" for feat, op, thresh in rule['conditions']]
            )
            lines.append(f"  IF {conditions_str}")
        else:
            lines.append("  IF (root node)")
        
        lines.append(f"  THEN credit_risk = {rule['prediction']}")
        lines.append(f"       Class distribution: {rule['class_distribution']}")
        lines.append("")
    
    return "\n".join(lines)


def rules_to_dataframe(rules: List[Dict]) -> pd.DataFrame:
    """Convert rules to pandas DataFrame."""
    rows = []
    for rule in rules:
        conditions_str = " AND ".join(
            [f"{feat} {op} {thresh:.4f}" for feat, op, thresh in rule['conditions']]
        )
        rows.append({
            'Rule_ID': rule['rule_id'],
            'Conditions': conditions_str,
            'Prediction': rule['prediction'],
            'Samples': rule['samples'],
            'Confidence': f"{rule['confidence']:.2%}",
            'Non_default': rule['class_distribution'].get('Non-default', 0),
            'Default': rule['class_distribution'].get('Default', 0)
        })
    
    return pd.DataFrame(rows)


def visualize_tree(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str] = ['Non-default', 'Default'],
    output_path: str = None,
    figsize: tuple = (20, 10)
):
    """
    Visualize the decision tree.
    
    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=True
    )
    plt.title("SHAP-KD Decision Tree", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Tree visualization saved to: {output_path}")
    
    plt.close()


def generate_latex_rules(rules: List[Dict], max_rules: int = 10) -> str:
    """
    Generate LaTeX-formatted rules for paper inclusion.
    
    Args:
        rules: List of rule dictionaries
        max_rules: Maximum number of rules to include
    
    Returns:
        LaTeX formatted string
    """
    # Sort by confidence and sample count
    sorted_rules = sorted(rules, key=lambda r: (-r['confidence'], -r['samples']))[:max_rules]
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Representative Decision Rules from SHAP-KD Model}")
    latex.append("  \\label{tab:extracted_rules}")
    latex.append("  \\begin{tabular}{llcc}")
    latex.append("    \\toprule")
    latex.append("    \\textbf{Rule ID} & \\textbf{Conditions} & \\textbf{Prediction} & \\textbf{Confidence} \\\\")
    latex.append("    \\midrule")
    
    for rule in sorted_rules:
        conditions = " $\\land$ ".join(
            [f"{feat.replace('_', '\\_')} {op} {thresh:.2f}" 
             for feat, op, thresh in rule['conditions'][:3]]  # Limit conditions for display
        )
        if len(rule['conditions']) > 3:
            conditions += " $\\land$ ..."
        
        latex.append(f"    {rule['rule_id']} & {conditions} & {rule['prediction']} & {rule['confidence']:.1%} \\\\")
    
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def export_rules_from_model_file(
    *,
    model_path: str,
    output_dir: str,
    max_rules: Optional[int] = None,
    visualize: bool = False,
    latex: bool = False,
) -> dict:
    """Load a saved student tree and export rule artifacts.

    The pickle is expected to contain at least: {'model': DecisionTreeClassifier}
    and may optionally contain 'feature_names'.
    """
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    tree = data['model']
    feature_names = data.get('feature_names', [f"Feature_{i}" for i in range(tree.n_features_in_)])

    rules = extract_rules_from_tree(tree, feature_names)
    text = rules_to_text(rules, max_rules)
    df = rules_to_dataframe(rules)

    out_txt = os.path.join(output_dir, 'rules.txt')
    out_xlsx = os.path.join(output_dir, 'rules.xlsx')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(text)
    df.to_excel(out_xlsx, index=False)

    out_png = None
    if visualize:
        out_png = os.path.join(output_dir, 'tree_visualization.png')
        visualize_tree(tree, feature_names, output_path=out_png)

    out_tex = None
    if latex:
        out_tex = os.path.join(output_dir, 'rules_latex.tex')
        tex = generate_latex_rules(rules, max_rules=10)
        with open(out_tex, 'w', encoding='utf-8') as f:
            f.write(tex)

    return {
        'rules': rules,
        'rules_txt': out_txt,
        'rules_xlsx': out_xlsx,
        'tree_png': out_png,
        'rules_tex': out_tex,
    }

