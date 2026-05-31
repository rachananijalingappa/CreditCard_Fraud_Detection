"""Feature engineering, scaling, train-test split, and SMOTE."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE, TEST_SIZE, PCA_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features: Hour and Amount_Log.
    
    Args:
        df: Dataset with 'Time' and 'Amount' columns.
        
    Returns:
        DataFrame with new features added.
    """
    df = df.copy()
    
    # Extract hour from time (if not already present)
    if 'Hour' not in df.columns:
        df['Hour'] = (df['Time'] / 3600).astype(int) % 24
    
    # Log transform amount to reduce skewness
    df['Amount_Log'] = np.log1p(df['Amount'])
    
    print('New features created: Hour, Amount_Log')
    print(f'Dataset shape: {df.shape}')
    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for training.
    
    Returns:
        List of feature column names.
    """
    return PCA_FEATURES + ['Amount', 'Time', 'Hour', 'Amount_Log']


def prepare_features(df: pd.DataFrame) -> tuple:
    """Separate features and target from the dataset.
    
    Args:
        df: Dataset with all features and 'Class' column.
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y = df['Class'].copy()
    
    print(f'Features shape: {X.shape}')
    print(f'NaN values: {X.isnull().sum().sum()}')
    print(f'Target distribution: {y.value_counts().to_dict()}')
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Split data into train and test sets with stratification.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print('Train-Test Split:')
    print(f'  Train: {X_train.shape[0]:,} samples')
    print(f'  Test:  {X_test.shape[0]:,} samples')
    print(f'\nTrain class distribution:')
    print(f'  Legitimate: {(y_train == 0).sum():,} | Fraud: {(y_train == 1).sum():,}')
    print(f'\nTest class distribution:')
    print(f'  Legitimate: {(y_test == 0).sum():,} | Fraud: {(y_test == 1).sum():,}')
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Apply StandardScaler - fit on training data ONLY to prevent data leakage.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler).
    """
    feature_cols = X_train.columns.tolist()
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )
    
    print('StandardScaler applied:')
    print('  - Fitted on training data only')
    print('  - Test data transformed using training statistics (no data leakage)')
    print(f'  - Train mean: {X_train_scaled.mean().mean():.6f} (should be ~0)')
    print(f'  - Train std:  {X_train_scaled.std().mean():.6f} (should be ~1)')
    
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Apply SMOTE on training data ONLY to handle class imbalance.
    
    Args:
        X_train: Scaled training features.
        y_train: Training target.
        
    Returns:
        Tuple of (X_train_smote, y_train_smote).
    """
    print('Before SMOTE:')
    print(f'  Legitimate: {(y_train == 0).sum():,}')
    print(f'  Fraud:      {(y_train == 1).sum():,}')
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f'\nAfter SMOTE:')
    print(f'  Legitimate: {(y_resampled == 0).sum():,}')
    print(f'  Fraud:      {(y_resampled == 1).sum():,}')
    print(f'  Synthetic fraud samples added: {(y_resampled == 1).sum() - (y_train == 1).sum():,}')
    
    return X_resampled, y_resampled
