"""Unit tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEngineerFeatures:
    """Tests for the engineer_features function."""
    
    def setup_method(self):
        """Create a sample DataFrame for testing."""
        self.df = pd.DataFrame({
            'Time': [0, 3600, 7200, 86400, 90000],
            'Amount': [100.0, 0.0, 50.5, 1000.0, 0.01],
            'Class': [0, 0, 1, 0, 1],
            'V1': [1, 2, 3, 4, 5],
        })
    
    def test_hour_column_created(self):
        """Hour column should be created from Time."""
        from src.preprocessing import engineer_features
        result = engineer_features(self.df)
        assert 'Hour' in result.columns
    
    def test_hour_values_correct(self):
        """Hour should be Time / 3600 mod 24."""
        from src.preprocessing import engineer_features
        result = engineer_features(self.df)
        expected_hours = [0, 1, 2, 0, 1]  # 0, 3600/3600=1, 7200/3600=2, 86400/3600=24%24=0, 90000/3600=25%24=1
        assert result['Hour'].tolist() == expected_hours
    
    def test_amount_log_created(self):
        """Amount_Log column should be created."""
        from src.preprocessing import engineer_features
        result = engineer_features(self.df)
        assert 'Amount_Log' in result.columns
    
    def test_amount_log_values(self):
        """Amount_Log should be log1p(Amount)."""
        from src.preprocessing import engineer_features
        result = engineer_features(self.df)
        expected = np.log1p(self.df['Amount'])
        np.testing.assert_array_almost_equal(result['Amount_Log'].values, expected.values)
    
    def test_does_not_modify_original(self):
        """Original DataFrame should not be modified."""
        from src.preprocessing import engineer_features
        original_cols = self.df.columns.tolist()
        engineer_features(self.df)
        assert self.df.columns.tolist() == original_cols
    
    def test_hour_not_duplicated(self):
        """If Hour already exists, it should not be overwritten."""
        from src.preprocessing import engineer_features
        self.df['Hour'] = [10, 10, 10, 10, 10]
        result = engineer_features(self.df)
        assert result['Hour'].tolist() == [10, 10, 10, 10, 10]


class TestScaleFeatures:
    """Tests for the scale_features function."""
    
    def setup_method(self):
        """Create sample train and test DataFrames."""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100) * 10 + 5,
        })
        self.X_test = pd.DataFrame({
            'A': np.random.randn(20),
            'B': np.random.randn(20) * 10 + 5,
        })
    
    def test_output_shapes(self):
        """Output shapes should match input shapes."""
        from src.preprocessing import scale_features
        X_train_s, X_test_s, scaler = scale_features(self.X_train, self.X_test)
        assert X_train_s.shape == self.X_train.shape
        assert X_test_s.shape == self.X_test.shape
    
    def test_train_mean_near_zero(self):
        """Scaled training data should have mean near 0."""
        from src.preprocessing import scale_features
        X_train_s, _, _ = scale_features(self.X_train, self.X_test)
        assert abs(X_train_s.mean().mean()) < 0.01
    
    def test_train_std_near_one(self):
        """Scaled training data should have std near 1."""
        from src.preprocessing import scale_features
        X_train_s, _, _ = scale_features(self.X_train, self.X_test)
        assert abs(X_train_s.std().mean() - 1.0) < 0.1
    
    def test_columns_preserved(self):
        """Column names should be preserved after scaling."""
        from src.preprocessing import scale_features
        X_train_s, X_test_s, _ = scale_features(self.X_train, self.X_test)
        assert X_train_s.columns.tolist() == ['A', 'B']
        assert X_test_s.columns.tolist() == ['A', 'B']
    
    def test_index_preserved(self):
        """Index should be preserved after scaling."""
        from src.preprocessing import scale_features
        X_train_s, X_test_s, _ = scale_features(self.X_train, self.X_test)
        assert X_train_s.index.tolist() == self.X_train.index.tolist()
        assert X_test_s.index.tolist() == self.X_test.index.tolist()
    
    def test_scaler_returned(self):
        """A fitted StandardScaler should be returned."""
        from src.preprocessing import scale_features
        _, _, scaler = scale_features(self.X_train, self.X_test)
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')


class TestApplySmote:
    """Tests for the apply_smote function."""
    
    def setup_method(self):
        """Create an imbalanced dataset."""
        np.random.seed(42)
        # 100 legitimate, 10 fraud (SMOTE needs at least k_neighbors+1 minority samples)
        X_legit = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
        X_fraud = pd.DataFrame(np.random.randn(10, 3) + 2, columns=['A', 'B', 'C'])
        self.X_train = pd.concat([X_legit, X_fraud], ignore_index=True)
        self.y_train = pd.Series([0] * 100 + [1] * 10)
    
    def test_classes_balanced(self):
        """After SMOTE, both classes should have equal counts."""
        from src.preprocessing import apply_smote
        X_res, y_res = apply_smote(self.X_train, self.y_train)
        assert (y_res == 0).sum() == (y_res == 1).sum()
    
    def test_majority_class_unchanged(self):
        """Majority class count should remain the same."""
        from src.preprocessing import apply_smote
        _, y_res = apply_smote(self.X_train, self.y_train)
        assert (y_res == 0).sum() == 100
    
    def test_minority_class_increased(self):
        """Minority class should have more samples after SMOTE."""
        from src.preprocessing import apply_smote
        _, y_res = apply_smote(self.X_train, self.y_train)
        assert (y_res == 1).sum() > 10


class TestSplitData:
    """Tests for the split_data function."""
    
    def setup_method(self):
        """Create a sample dataset."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
        self.y = pd.Series([0] * 900 + [1] * 100)
    
    def test_split_sizes(self):
        """Default 80/20 split should produce correct sizes."""
        from src.preprocessing import split_data
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        assert len(X_train) == 800
        assert len(X_test) == 200
    
    def test_stratification(self):
        """Both splits should preserve class proportions."""
        from src.preprocessing import split_data
        _, _, y_train, y_test = split_data(self.X, self.y)
        train_ratio = (y_train == 1).mean()
        test_ratio = (y_test == 1).mean()
        assert abs(train_ratio - 0.10) < 0.02
        assert abs(test_ratio - 0.10) < 0.02


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
