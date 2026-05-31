"""Model evaluation, comparison, and explainability."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve,
                             average_precision_score, auc)

from src.config import RANDOM_STATE


def create_comparison_table(all_results: list) -> pd.DataFrame:
    """Create a sorted model comparison DataFrame.
    
    Args:
        all_results: List of result dictionaries from evaluate().
        
    Returns:
        Sorted DataFrame with all model metrics.
    """
    results_df = pd.DataFrame(all_results).sort_values(
        'F1-Score', ascending=False
    ).reset_index(drop=True)
    
    print('MODEL COMPARISON (sorted by F1-Score)')
    print('=' * 100)
    display(results_df)
    
    best_model = results_df.iloc[0]['Model']
    print(f'\nBest model: {best_model}')
    
    return results_df


def plot_confusion_matrices(all_preds: dict, y_test) -> None:
    """Plot confusion matrices for all models in a grid."""
    n_models = len(all_preds)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, (name, preds) in enumerate(all_preds.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(all_probs: dict, y_test) -> None:
    """Plot ROC curves for all models."""
    plt.figure(figsize=(12, 8))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pr_curves(all_probs: dict, y_test) -> None:
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(12, 8))
    for name, probs in all_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        plt.plot(rec, prec, linewidth=2, label=f'{name} (AP={ap:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Plot bar chart comparing model metrics."""
    metrics_plot = ['Precision', 'Recall', 'F1-Score', 'MCC']
    plot_data = results_df[['Model'] + metrics_plot].set_index('Model')
    
    plot_data.plot(kind='bar', figsize=(16, 8), width=0.8, edgecolor='black')
    plt.title('Model Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_shap_analysis(model, X_test: pd.DataFrame, n_samples: int = 2000) -> None:
    """Run SHAP analysis on a tree-based model."""
    print('SHAP Analysis...')
    
    shap_sample = X_test.sample(min(n_samples, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    
    # Feature importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_sample, plot_type='bar', show=False, max_display=15)
    plt.title('SHAP Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Detailed impact
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=15)
    plt.title('SHAP Feature Impact', fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_best_model_report(results_df: pd.DataFrame, all_preds: dict, y_test) -> None:
    """Print classification report and summary for the best model."""
    best_model = results_df.iloc[0]['Model']
    best_preds = all_preds.get(best_model, all_preds['XGBoost (Tuned)'])
    
    print(f'Classification Report - {best_model}')
    print('=' * 60)
    print(classification_report(y_test, best_preds, target_names=['Legitimate', 'Fraud']))


def print_final_summary(results_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """Print the final project summary."""
    best_model = results_df.iloc[0]['Model']
    best = results_df.iloc[0]
    
    print('=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    print(f'\nDataset: {df.shape[0]:,} transactions, {(df["Class"]==1).sum()} fraud ({df["Class"].mean()*100:.3f}%)')
    print(f'Models trained: {len(results_df)}')
    print(f'Best model: {best_model}')
    print(f'\nBest model metrics:')
    print(f'  Precision: {best["Precision"]:.4f}')
    print(f'  Recall:    {best["Recall"]:.4f}')
    print(f'  F1-Score:  {best["F1-Score"]:.4f}')
    print(f'\nKey findings:')
    print(f'  - Accuracy alone is misleading for imbalanced data')
    print(f'  - Recall is critical: catching fraud matters more than avoiding false alarms')
    print(f'  - SMOTE on training data only prevents data leakage')
    print(f'  - Ensemble models (XGBoost/LightGBM) perform best on tabular data')
    print(f'\nFuture work:')
    print(f'  - Deploy as a REST API for real-time predictions')
    print(f'  - Explore federated learning for privacy-preserving training')
    print(f'  - Implement real-time monitoring with model drift detection')
