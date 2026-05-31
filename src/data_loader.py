"""Data loading, cleaning, and initial inspection."""

import numpy as np
import pandas as pd
from src.config import DATA_PATH


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the loaded dataset.
    """
    df = pd.read_csv(path)
    print(f'Dataset shape: {df.shape}')
    print(f'Rows: {df.shape[0]:,} | Columns: {df.shape[1]}')
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print basic dataset info, check for issues.

    Checks for:
    - Missing values
    - Duplicate rows (removes them if found)
    - Infinite values (replaces with median)
    - Class distribution

    Args:
        df: The dataset DataFrame.

    Returns:
        DataFrame after cleaning.
    """
    # Basic info
    print('Dataset Info:')
    print('=' * 50)
    df.info()
    print('\nStatistical Summary:')
    display(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing duplicates and handling infinities.

    Args:
        df: Raw dataset.

    Returns:
        Cleaned DataFrame.
    """
    # Check missing values
    print('Missing values per column:')
    print(df.isnull().sum().sum())
    print('No missing values found!' if df.isnull().sum().sum() == 0 else 'Missing values detected!')

    # Remove duplicates
    duplicates = df.duplicated().sum()
    print(f'Duplicate rows: {duplicates:,}')
    if duplicates > 0:
        print(f'Removing {duplicates} duplicates...')
        df = df.drop_duplicates().reset_index(drop=True)
        print(f'New shape: {df.shape}')

    # Handle infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f'Infinite values: {inf_count}')
    if inf_count > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(), inplace=True)
        print('Infinite values replaced with median.')

    return df


def print_class_distribution(df: pd.DataFrame) -> None:
    """Print class distribution statistics.

    Args:
        df: Dataset with 'Class' column.
    """
    class_counts = df['Class'].value_counts()
    class_pct = df['Class'].value_counts(normalize=True) * 100

    print('Class Distribution:')
    print(f'  Legitimate (0): {class_counts[0]:,} ({class_pct[0]:.3f}%)')
    print(f'  Fraud (1):      {class_counts[1]:,} ({class_pct[1]:.3f}%)')
    print(f'\nImbalance ratio: 1:{class_counts[0] // class_counts[1]}')
    print('This is extremely imbalanced - accuracy alone will be misleading!')


def compare_fraud_stats(df: pd.DataFrame) -> None:
    """Compare statistics of fraud vs legitimate transactions.

    Args:
        df: Dataset with 'Class', 'Amount', 'Time' columns.
    """
    print('Legitimate Transactions:')
    display(df[df['Class'] == 0][['Amount', 'Time']].describe())
    print('\nFraudulent Transactions:')
    display(df[df['Class'] == 1][['Amount', 'Time']].describe())
