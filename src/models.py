"""Classical ML and ensemble model training with Optuna hyperparameter tuning."""

import time
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score)

from src.config import RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate(name: str, y_true, y_pred, y_prob=None, training_time: float = None,
             all_results: list = None, all_preds: dict = None, all_probs: dict = None) -> dict:
    """Evaluate a model and store results.
    
    Args:
        name: Model name.
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional).
        training_time: Time taken to train (optional).
        all_results: List to append results to.
        all_preds: Dict to store predictions.
        all_probs: Dict to store probabilities.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    result = {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        result['ROC-AUC'] = roc_auc_score(y_true, y_prob)
        result['AUPRC'] = average_precision_score(y_true, y_prob)
    if training_time:
        result['Time(s)'] = round(training_time, 2)
    
    if all_results is not None:
        all_results.append(result)
    if all_preds is not None:
        all_preds[name] = y_pred
    if y_prob is not None and all_probs is not None:
        all_probs[name] = y_prob
    
    print(f'{name}:')
    print(f'  Accuracy={result["Accuracy"]:.4f}  Precision={result["Precision"]:.4f}  '
          f'Recall={result["Recall"]:.4f}  F1={result["F1-Score"]:.4f}  MCC={result["MCC"]:.4f}')
    if y_prob is not None:
        print(f'  ROC-AUC={result["ROC-AUC"]:.4f}  AUPRC={result["AUPRC"]:.4f}')
    
    return result


def train_classical_models(X_train, y_train, X_test, y_test,
                           all_results, all_preds, all_probs) -> dict:
    """Train classical ML models: LR, DT, RF, SVM, KNN.
    
    Returns:
        Dictionary of trained model objects.
    """
    models = {}
    
    # Logistic Regression
    start = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_train, y_train)
    t = time.time() - start
    evaluate('Logistic Regression', y_test, lr.predict(X_test),
             lr.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    models['lr'] = lr
    
    # Decision Tree
    start = time.time()
    dt = DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    t = time.time() - start
    evaluate('Decision Tree', y_test, dt.predict(X_test),
             dt.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    models['dt'] = dt
    
    # Random Forest
    start = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    t = time.time() - start
    evaluate('Random Forest', y_test, rf.predict(X_test),
             rf.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    models['rf'] = rf
    
    # SVM (subset for speed)
    sample_n = min(50000, len(X_train))
    idx = np.random.choice(len(X_train), sample_n, replace=False)
    start = time.time()
    svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train.iloc[idx], y_train.iloc[idx])
    t = time.time() - start
    evaluate('SVM', y_test, svm.predict(X_test),
             svm.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    print(f'  (Trained on {sample_n:,} samples for speed)')
    models['svm'] = svm
    
    # KNN
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    t = time.time() - start
    evaluate('KNN', y_test, knn.predict(X_test),
             knn.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    models['knn'] = knn
    
    return models


def train_ensemble_models(X_train, y_train, X_test, y_test,
                          all_results, all_preds, all_probs) -> dict:
    """Train ensemble models: XGBoost and LightGBM.
    
    Returns:
        Dictionary of trained model objects.
    """
    models = {}
    
    # XGBoost
    start = time.time()
    xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                        random_state=RANDOM_STATE, eval_metric='logloss',
                        n_jobs=-1, use_label_encoder=False)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    t = time.time() - start
    evaluate('XGBoost', y_test, xgb.predict(X_test),
             xgb.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    models['xgb'] = xgb
    
    # LightGBM
    start = time.time()
    lgbm = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgbm.fit(X_train, y_train)
    t = time.time() - start
    evaluate('LightGBM', y_test, lgbm.predict(X_test),
             lgbm.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    models['lgbm'] = lgbm
    
    return models


def tune_xgboost(X_train, y_train, X_test, y_test,
                 all_results, all_preds, all_probs, n_trials: int = 30) -> XGBClassifier:
    """Tune XGBoost with Optuna and train the best model.
    
    Returns:
        Tuned XGBClassifier.
    """
    print(f'Tuning XGBoost with Optuna ({n_trials} trials)...')
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }
        model = XGBClassifier(**params, random_state=RANDOM_STATE,
                              eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f'Best CV F1: {study.best_value:.4f}')
    
    # Train best model
    start = time.time()
    xgb_tuned = XGBClassifier(**study.best_params, random_state=RANDOM_STATE,
                              eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
    xgb_tuned.fit(X_train, y_train)
    t = time.time() - start
    
    evaluate('XGBoost (Tuned)', y_test, xgb_tuned.predict(X_test),
             xgb_tuned.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    
    return xgb_tuned


def tune_lightgbm(X_train, y_train, X_test, y_test,
                  all_results, all_preds, all_probs, n_trials: int = 30) -> LGBMClassifier:
    """Tune LightGBM with Optuna and train the best model.
    
    Returns:
        Tuned LGBMClassifier.
    """
    print(f'Tuning LightGBM with Optuna ({n_trials} trials)...')
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        }
        model = LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f'Best CV F1: {study.best_value:.4f}')
    
    # Train best model
    start = time.time()
    lgbm_tuned = LGBMClassifier(**study.best_params, random_state=RANDOM_STATE,
                                n_jobs=-1, verbose=-1)
    lgbm_tuned.fit(X_train, y_train)
    t = time.time() - start
    
    evaluate('LightGBM (Tuned)', y_test, lgbm_tuned.predict(X_test),
             lgbm_tuned.predict_proba(X_test)[:, 1], t, all_results, all_preds, all_probs)
    
    return lgbm_tuned
