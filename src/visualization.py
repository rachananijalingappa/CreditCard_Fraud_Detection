"""Exploratory Data Analysis visualizations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE, COLORS, PCA_FEATURES


# ── Class Distribution ──────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot class distribution as bar chart and pie chart.
    
    Args:
        df: Dataset with 'Class' column.
    """
    class_counts = df['Class'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    bars = axes[0].bar(['Legitimate', 'Fraud'], class_counts.values,
                       color=COLORS, edgecolor='black')
    axes[0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.3f%%',
                colors=COLORS, startangle=90, explode=(0, 0.1), shadow=True)
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ── Amount Distribution ─────────────────────────────────────────────────────

def plot_amount_distribution(df: pd.DataFrame) -> None:
    """Plot transaction amount distribution by class.
    
    Args:
        df: Dataset with 'Amount' and 'Class' columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Histogram
    axes[0].hist(df[df['Class'] == 0]['Amount'], bins=100, alpha=0.7,
                 color='#2ecc71', label='Legitimate', density=True)
    axes[0].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7,
                 color='#e74c3c', label='Fraud', density=True)
    axes[0].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Amount')
    axes[0].set_xlim(0, 500)
    axes[0].legend()
    
    # Box plot
    df_plot = df[['Amount', 'Class']].copy()
    df_plot['Class'] = df_plot['Class'].map({0: 'Legitimate', 1: 'Fraud'})
    sns.boxplot(data=df_plot, x='Class', y='Amount', palette=COLORS,
                ax=axes[1], showfliers=False)
    axes[1].set_title('Amount by Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f'Legitimate avg amount: ${df[df["Class"]==0]["Amount"].mean():.2f}')
    print(f'Fraud avg amount:      ${df[df["Class"]==1]["Amount"].mean():.2f}')


# ── Time Distribution ────────────────────────────────────────────────────────

def plot_time_distribution(df: pd.DataFrame) -> None:
    """Plot transaction time distribution and fraud by hour.
    
    Note: Creates 'Hour' column if not already present.
    
    Args:
        df: Dataset with 'Time' and 'Class' columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Time by class
    axes[0].hist(df[df['Class'] == 0]['Time'], bins=100, alpha=0.7,
                 color='#2ecc71', label='Legitimate', density=True)
    axes[0].hist(df[df['Class'] == 1]['Time'], bins=100, alpha=0.7,
                 color='#e74c3c', label='Fraud', density=True)
    axes[0].set_title('Transaction Time Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].legend()
    
    # Ensure Hour column exists
    if 'Hour' not in df.columns:
        df['Hour'] = (df['Time'] / 3600).astype(int) % 24
    
    # Fraud by hour
    fraud_by_hour = df[df['Class'] == 1].groupby('Hour').size()
    axes[1].bar(fraud_by_hour.index, fraud_by_hour.values, color='#e74c3c', alpha=0.8)
    axes[1].set_title('Fraud Transactions by Hour', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Fraud Count')
    
    plt.tight_layout()
    plt.show()


# ── Correlation Heatmap ──────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> pd.Series:
    """Plot correlation heatmap and return target correlations.
    
    Args:
        df: Full dataset.
        
    Returns:
        Series of absolute correlations with 'Class', sorted descending.
    """
    plt.figure(figsize=(20, 16))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                linewidths=0.5, square=True)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Top features correlated with fraud
    target_corr = corr_matrix['Class'].drop('Class').abs().sort_values(ascending=False)
    print('Top 10 features correlated with fraud:')
    for feat, val in target_corr.head(10).items():
        print(f'  {feat}: {corr_matrix.loc[feat, "Class"]:.4f}')
    
    return target_corr


# ── Violin Plots ─────────────────────────────────────────────────────────────

def plot_violin_features(df: pd.DataFrame, target_corr: pd.Series, n_features: int = 8) -> None:
    """Plot violin plots for top discriminative features.
    
    Args:
        df: Full dataset.
        target_corr: Sorted correlation series from plot_correlation_heatmap().
        n_features: Number of top features to plot.
    """
    top_features = target_corr.head(n_features).index.tolist()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_features):
        temp = df[[feat, 'Class']].copy()
        temp['Class'] = temp['Class'].map({0: 'Legitimate', 1: 'Fraud'})
        sns.violinplot(data=temp, x='Class', y=feat, palette=COLORS,
                       ax=axes[i], inner='box', cut=0)
        axes[i].set_title(feat, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
    
    plt.suptitle('Top 8 Discriminative Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# ── Outlier Analysis ─────────────────────────────────────────────────────────

def analyze_outliers(df: pd.DataFrame, target_corr: pd.Series) -> None:
    """Perform outlier analysis using IQR method.
    
    Args:
        df: Full dataset.
        target_corr: Sorted correlation series.
    """
    top_features = target_corr.head(6).index.tolist()
    
    outlier_results = []
    for col in ['Amount', 'Time'] + top_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_results.append({
            'Feature': col, 'Outliers': outliers,
            'Percentage': round(outliers / len(df) * 100, 2)
        })
    
    print('Outlier Analysis (IQR Method):')
    display(pd.DataFrame(outlier_results))
    print('\nNote: Outliers in PCA features may indicate fraud patterns, so we keep them.')


# ── PCA Analysis ─────────────────────────────────────────────────────────────

def plot_pca_variance(df: pd.DataFrame) -> None:
    """Plot variance analysis of PCA components.
    
    Args:
        df: Dataset with V1-V28 columns.
    """
    pca_variances = df[PCA_FEATURES].var().sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Variance per component
    axes[0].bar(range(1, 29), pca_variances.values, color='#3498db',
                edgecolor='black', linewidth=0.5)
    axes[0].set_title('Variance of PCA Components', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Variance')
    
    # Cumulative variance
    cum_var = np.cumsum(pca_variances.values / pca_variances.sum())
    axes[1].plot(range(1, 29), cum_var, 'ro-')
    axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
    axes[1].set_title('Cumulative Variance', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Components')
    axes[1].set_ylabel('Cumulative Proportion')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    n_95 = np.argmax(cum_var >= 0.95) + 1
    print(f'{n_95} components explain 95% of the variance.')


def plot_pca_2d(df: pd.DataFrame) -> np.ndarray:
    """Create 2D PCA projection plot.
    
    Args:
        df: Full dataset.
        
    Returns:
        2D projected array (used by clustering later).
    """
    all_features = PCA_FEATURES + ['Amount', 'Time']
    scaler_pca = StandardScaler()
    X_pca_input = scaler_pca.fit_transform(df[all_features])
    
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_pca_input)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(X_2d[df['Class'] == 0, 0], X_2d[df['Class'] == 0, 1],
                c='#2ecc71', alpha=0.1, s=3, label='Legitimate')
    plt.scatter(X_2d[df['Class'] == 1, 0], X_2d[df['Class'] == 1, 1],
                c='#e74c3c', alpha=0.8, s=15, label='Fraud',
                edgecolors='black', linewidths=0.3)
    plt.title('2D PCA Projection', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.show()
    
    return X_2d


def plot_pca_3d(df: pd.DataFrame) -> None:
    """Create interactive 3D PCA projection plot.
    
    Args:
        df: Full dataset.
    """
    all_features = PCA_FEATURES + ['Amount', 'Time']
    scaler_pca = StandardScaler()
    X_pca_input = scaler_pca.fit_transform(df[all_features])
    
    pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
    X_3d = pca_3d.fit_transform(X_pca_input)
    
    # Sample for performance
    legit_idx = np.random.choice(np.where(df['Class'] == 0)[0], 5000, replace=False)
    fraud_idx = np.where(df['Class'] == 1)[0]
    sample_idx = np.concatenate([legit_idx, fraud_idx])
    
    fig = px.scatter_3d(x=X_3d[sample_idx, 0], y=X_3d[sample_idx, 1],
                        z=X_3d[sample_idx, 2],
                        color=df['Class'].iloc[sample_idx].astype(str),
                        color_discrete_map={'0': '#2ecc71', '1': '#e74c3c'},
                        title='3D PCA Projection (Sampled)', opacity=0.6)
    fig.update_layout(template='plotly_dark', width=900, height=600)
    fig.show()


def plot_feature_importance(df: pd.DataFrame, corr_matrix: pd.DataFrame) -> None:
    """Plot feature importance using correlation with fraud.
    
    Args:
        df: Full dataset.
        corr_matrix: Correlation matrix of the dataset.
    """
    all_features = PCA_FEATURES + ['Amount', 'Time']
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    
    importance = []
    for col in all_features:
        stat, p_val = stats.mannwhitneyu(fraud[col], legit[col], alternative='two-sided')
        importance.append({
            'Feature': col, 'P-Value': p_val,
            'Correlation': abs(corr_matrix.loc[col, 'Class'])
        })
    
    importance_df = pd.DataFrame(importance).sort_values('Correlation', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['Feature'].head(15)[::-1],
             importance_df['Correlation'].head(15)[::-1],
             color='#3498db', edgecolor='black')
    plt.title('Feature Importance (Correlation with Fraud)', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.show()


# ── SMOTE Visualization ─────────────────────────────────────────────────────

def plot_smote_effect(y_train, y_train_smote) -> None:
    """Visualize the effect of SMOTE on class balance.
    
    Args:
        y_train: Original training labels.
        y_train_smote: SMOTE-resampled training labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    before = [y_train.value_counts()[0], y_train.value_counts()[1]]
    bars1 = axes[0].bar(['Legitimate', 'Fraud'], before, color=COLORS, edgecolor='black')
    axes[0].set_title('Before SMOTE', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    for bar, c in zip(bars1, before):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{c:,}', ha='center', va='bottom', fontweight='bold')
    
    # After
    after = [y_train_smote.value_counts()[0], y_train_smote.value_counts()[1]]
    bars2 = axes[1].bar(['Legitimate', 'Fraud'], after, color=COLORS, edgecolor='black')
    axes[1].set_title('After SMOTE', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    for bar, c in zip(bars2, after):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{c:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Effect of SMOTE on Class Balance', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()
