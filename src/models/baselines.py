#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline Models for Credit Scoring

Categories:
    - Linear: LR-Ridge, LR-Lasso, LR-ElasticNet
    - Kernel: SVM-RBF, SVM-Linear
    - Ensemble: RF, GBDT, XGBoost, LightGBM, CatBoost
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    model: Any
    description: str
    reference: str
    category: str


def get_baseline_models(use_gpu: bool = True, random_state: int = 42) -> Dict[str, ModelConfig]:
    """Get all baseline models."""
    
    xgb_device = 'cuda' if use_gpu else 'cpu'
    lgb_device = 'gpu' if use_gpu else 'cpu'
    cat_task_type = 'GPU' if use_gpu else 'CPU'
    
    models = {
        # Linear Models
        'LR-Ridge': ModelConfig(
            name='LR-Ridge',
            model=LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            ),
            description='Logistic Regression with L2',
            reference='Hosmer & Lemeshow (2000)',
            category='Linear'
        ),
        
        'LR-Lasso': ModelConfig(
            name='LR-Lasso',
            model=LogisticRegression(
                penalty='l1',
                solver='saga',
                max_iter=1000,
                random_state=random_state
            ),
            description='Logistic Regression with L1 Regularization (Lasso)',
            reference='Tibshirani (1996)',
            category='Linear'
        ),
        
        'LR-ElasticNet': ModelConfig(
            name='LR-ElasticNet',
            model=LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.5,
                max_iter=1000,
                random_state=random_state
            ),
            description='Logistic Regression with Elastic Net Regularization',
            reference='Zou & Hastie (2005)',
            category='Linear'
        ),
        
        # =====================================================================
        # Kernel Methods
        # =====================================================================
        'SVM-RBF': ModelConfig(
            name='SVM-RBF',
            model=SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state
            ),
            description='Support Vector Machine with RBF Kernel',
            reference='Cortes & Vapnik (1995)',
            category='Kernel'
        ),
        
        'SVM-Linear': ModelConfig(
            name='SVM-Linear',
            model=SVC(
                kernel='linear',
                probability=True,
                random_state=random_state
            ),
            description='Support Vector Machine with Linear Kernel',
            reference='Cortes & Vapnik (1995)',
            category='Kernel'
        ),
        
        # =====================================================================
        # Tree-Based Models
        # =====================================================================
        'DT': ModelConfig(
            name='DT',
            model=DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=random_state
            ),
            description='Decision Tree Classifier',
            reference='Breiman et al. (1984)',
            category='Tree'
        ),
        
        'RF': ModelConfig(
            name='RF',
            model=RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1,
                random_state=random_state
            ),
            description='Random Forest Classifier',
            reference='Breiman (2001)',
            category='Ensemble'
        ),
        
        'GBDT': ModelConfig(
            name='GBDT',
            model=GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state
            ),
            description='Gradient Boosting Decision Tree',
            reference='Friedman (2001)',
            category='Ensemble'
        ),
        
        'XGBoost': ModelConfig(
            name='XGBoost',
            model=xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                device=xgb_device,
                random_state=random_state
            ),
            description='Extreme Gradient Boosting',
            reference='Chen & Guestrin (2016)',
            category='Ensemble'
        ),
        
        'LightGBM': ModelConfig(
            name='LightGBM',
            model=lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                device=lgb_device,
                verbose=-1,
                random_state=random_state
            ),
            description='Light Gradient Boosting Machine',
            reference='Ke et al. (2017)',
            category='Ensemble'
        ),
        
        'CatBoost': ModelConfig(
            name='CatBoost',
            model=CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                task_type=cat_task_type,
                devices='0' if use_gpu else None,
                verbose=False,
                allow_writing_files=False,
                random_state=random_state
            ),
            description='Categorical Boosting',
            reference='Prokhorenkova et al. (2018)',
            category='Ensemble'
        ),
        
        # =====================================================================
        # Other Classical Models
        # =====================================================================
        'KNN': ModelConfig(
            name='KNN',
            model=KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            description='K-Nearest Neighbors Classifier',
            reference='Cover & Hart (1967)',
            category='Instance-Based'
        ),
        
        'NB': ModelConfig(
            name='NB',
            model=GaussianNB(),
            description='Gaussian Naive Bayes',
            reference='Hand & Yu (2001)',
            category='Probabilistic'
        ),
    }
    
    return models


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
        metrics['ap'] = average_precision_score(y_true, y_proba)
        metrics['brier'] = brier_score_loss(y_true, y_proba)
    
    return metrics


def run_baseline_experiments(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             models: Dict[str, ModelConfig] = None,
                             use_gpu: bool = True) -> Dict[str, Dict]:
    """
    Run experiments with all baseline models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models: Dictionary of models (uses default if None)
        use_gpu: Whether to use GPU
        
    Returns:
        Dictionary mapping model names to their results
    """
    if models is None:
        models = get_baseline_models(use_gpu=use_gpu)
    
    results = {}
    
    print("=" * 70)
    print("Running Baseline Experiments")
    print("=" * 70)
    print(f"{'Model':<15} {'AUC':<10} {'Acc':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    
    for name, config in models.items():
        try:
            # Train model
            model = config.model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_proba)
            metrics['model_name'] = name
            metrics['reference'] = config.reference
            metrics['category'] = config.category
            
            results[name] = metrics
            
            # Print results
            auc_str = f"{metrics.get('auc', 0):.4f}" if 'auc' in metrics else "N/A"
            print(f"{name:<15} {auc_str:<10} {metrics['accuracy']:.4f}     "
                  f"{metrics['f1']:.4f}     {metrics['precision']:.4f}     {metrics['recall']:.4f}")
            
        except Exception as e:
            print(f"{name:<15} ERROR: {str(e)[:50]}")
            results[name] = {'error': str(e)}
    
    print("=" * 70)
    
    return results


# =============================================================================
# Optuna Hyperparameter Optimization
# =============================================================================

def optimize_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           n_trials: int = 50,
                           random_state: int = 42) -> Dict:
    """
    Optimize Random Forest hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        random_state: Random seed
        
    Returns:
        Dictionary with best_params and best_score
    """
    if not OPTUNA_AVAILABLE:
        print("[WARNING] Optuna not available. Using default parameters.")
        return {'best_params': {}, 'best_score': 0.0}
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'n_jobs': -1,
            'random_state': random_state
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_proba)
        
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }


def optimize_gbdt(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  n_trials: int = 50,
                  random_state: int = 42) -> Dict:
    """
    Optimize GBDT (GradientBoostingClassifier) hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        random_state: Random seed
        
    Returns:
        Dictionary with best_params and best_score
    """
    if not OPTUNA_AVAILABLE:
        print("[WARNING] Optuna not available. Using default parameters.")
        return {'best_params': {}, 'best_score': 0.0}
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': random_state
        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_proba)
        
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }


def optimize_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     n_trials: int = 50, use_gpu: bool = True,
                     random_state: int = 42) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        use_gpu: Whether to use GPU
        random_state: Random seed
        
    Returns:
        Dictionary with best_params and best_score
    """
    if not OPTUNA_AVAILABLE:
        print("[WARNING] Optuna not available. Using default parameters.")
        return {'best_params': {}, 'best_score': 0.0}
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu else 'cpu',
            'random_state': random_state
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_proba)
        
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }


def optimize_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      n_trials: int = 50, use_gpu: bool = True,
                      random_state: int = 42) -> Dict:
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        use_gpu: Whether to use GPU
        random_state: Random seed
        
    Returns:
        Dictionary with best_params and best_score
    """
    if not OPTUNA_AVAILABLE:
        print("[WARNING] Optuna not available. Using default parameters.")
        return {'best_params': {}, 'best_score': 0.0}
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'device': 'gpu' if use_gpu else 'cpu',
            'verbose': -1,
            'random_state': random_state
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        y_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_proba)
        
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }


def optimize_catboost(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      n_trials: int = 50, use_gpu: bool = True,
                      random_state: int = 42) -> Dict:
    """
    Optimize CatBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        use_gpu: Whether to use GPU
        random_state: Random seed
        
    Returns:
        Dictionary with best_params and best_score
    """
    if not OPTUNA_AVAILABLE:
        print("[WARNING] Optuna not available. Using default parameters.")
        return {'best_params': {}, 'best_score': 0.0}
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'task_type': 'GPU' if use_gpu else 'CPU',
            'devices': '0' if use_gpu else None,
            'verbose': False,
            'allow_writing_files': False,
            'random_state': random_state
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_proba)
        
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': len(study.trials)
    }


def get_optimized_boosting_models(X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   n_trials: int = 50, use_gpu: bool = True,
                                   random_state: int = 42) -> Tuple[Dict[str, ModelConfig], Dict]:
    """
    Get all ensemble models optimized with Optuna.
    
    Optimizes: Random Forest, GBDT, XGBoost, LightGBM, CatBoost
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials per model
        use_gpu: Whether to use GPU for applicable models
        random_state: Random seed
        
    Returns:
        Tuple of (models_dict, optimization_results_dict)
    """
    # Optimize Random Forest
    print("[INFO] Optimizing Random Forest...")
    rf_result = optimize_random_forest(X_train, y_train, X_val, y_val, n_trials, random_state)
    rf_params = rf_result['best_params'].copy()
    rf_params.update({'n_jobs': -1, 'random_state': random_state})
    print(f"  Best AUC: {rf_result['best_score']:.4f}")
    
    # Optimize GBDT
    print("[INFO] Optimizing GBDT...")
    gbdt_result = optimize_gbdt(X_train, y_train, X_val, y_val, n_trials, random_state)
    gbdt_params = gbdt_result['best_params'].copy()
    gbdt_params.update({'random_state': random_state})
    print(f"  Best AUC: {gbdt_result['best_score']:.4f}")
    
    # Optimize XGBoost
    print("[INFO] Optimizing XGBoost...")
    xgb_result = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials, use_gpu, random_state)
    xgb_params = xgb_result['best_params'].copy()
    xgb_params.update({
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu',
        'random_state': random_state
    })
    print(f"  Best AUC: {xgb_result['best_score']:.4f}")
    
    # Optimize LightGBM
    print("[INFO] Optimizing LightGBM...")
    lgb_result = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials, use_gpu, random_state)
    lgb_params = lgb_result['best_params'].copy()
    lgb_params.update({
        'device': 'gpu' if use_gpu else 'cpu',
        'verbose': -1,
        'random_state': random_state
    })
    print(f"  Best AUC: {lgb_result['best_score']:.4f}")
    
    # Optimize CatBoost
    print("[INFO] Optimizing CatBoost...")
    cat_result = optimize_catboost(X_train, y_train, X_val, y_val, n_trials, use_gpu, random_state)
    cat_params = cat_result['best_params'].copy()
    cat_params.update({
        'task_type': 'GPU' if use_gpu else 'CPU',
        'devices': '0' if use_gpu else None,
        'verbose': False,
        'allow_writing_files': False,
        'random_state': random_state
    })
    print(f"  Best AUC: {cat_result['best_score']:.4f}")
    
    models = {
        'RF-Tuned': ModelConfig(
            name='RF-Tuned',
            model=RandomForestClassifier(**rf_params),
            description=f'Random Forest (Optuna optimized, {n_trials} trials)',
            reference='Breiman (2001) + Optuna',
            category='Ensemble-Optimized'
        ),
        'GBDT-Tuned': ModelConfig(
            name='GBDT-Tuned',
            model=GradientBoostingClassifier(**gbdt_params),
            description=f'GBDT (Optuna optimized, {n_trials} trials)',
            reference='Friedman (2001) + Optuna',
            category='Ensemble-Optimized'
        ),
        'XGBoost-Tuned': ModelConfig(
            name='XGBoost-Tuned',
            model=xgb.XGBClassifier(**xgb_params),
            description=f'XGBoost (Optuna optimized, {n_trials} trials)',
            reference='Chen & Guestrin (2016) + Optuna',
            category='Ensemble-Optimized'
        ),
        'LightGBM-Tuned': ModelConfig(
            name='LightGBM-Tuned',
            model=lgb.LGBMClassifier(**lgb_params),
            description=f'LightGBM (Optuna optimized, {n_trials} trials)',
            reference='Ke et al. (2017) + Optuna',
            category='Ensemble-Optimized'
        ),
        'CatBoost-Tuned': ModelConfig(
            name='CatBoost-Tuned',
            model=CatBoostClassifier(**cat_params),
            description=f'CatBoost (Optuna optimized, {n_trials} trials)',
            reference='Prokhorenkova et al. (2018) + Optuna',
            category='Ensemble-Optimized'
        )
    }
    
    # Train each model and store in optimization_results with model instance
    trained_models = {}
    for model_name, config in models.items():
        model_copy = config.model.__class__(**config.model.get_params())
        model_copy.fit(X_train, y_train)
        trained_models[model_name] = model_copy
    
    optimization_results = {
        'RF-Tuned': {**rf_result, 'best_model': trained_models.get('RF-Tuned')},
        'GBDT-Tuned': {**gbdt_result, 'best_model': trained_models.get('GBDT-Tuned')},
        'XGBoost-Tuned': {**xgb_result, 'best_model': trained_models.get('XGBoost-Tuned')},
        'LightGBM-Tuned': {**lgb_result, 'best_model': trained_models.get('LightGBM-Tuned')},
        'CatBoost-Tuned': {**cat_result, 'best_model': trained_models.get('CatBoost-Tuned')}
    }
    
    # Find best model
    best_model = max(optimization_results.items(), key=lambda x: x[1]['best_score'])
    print(f"\n[BEST] {best_model[0].upper()}: AUC = {best_model[1]['best_score']:.4f}")
    
    return models, optimization_results


def train_traditional_baselines(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                random_state: int = 42) -> Dict[str, Dict]:
    """
    Train traditional baseline models (Linear, Kernel, etc.) without Optuna optimization.
    
    These are simpler models that don't require extensive hyperparameter tuning:
    - LR-Ridge, LR-Lasso, LR-ElasticNet (Linear)
    - SVM-RBF, SVM-Linear (Kernel)
    - DT (Decision Tree)
    - KNN, NB (Instance-based, Probabilistic)
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed
        
    Returns:
        Dictionary mapping model names to (model, val_auc, hyperparams)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    
    print("[INFO] Training traditional baseline models...")
    
    traditional_models = {
        # Linear Models
        'LR-Ridge': LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=1000, random_state=random_state
        ),
        'LR-Lasso': LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, random_state=random_state
        ),
        'LR-ElasticNet': LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=random_state
        ),
        # Kernel Methods
        'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=random_state),
        'SVM-Linear': SVC(kernel='linear', probability=True, random_state=random_state),
        # Tree
        'DT': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=random_state),
        # Instance-based
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        # Probabilistic
        'NB': GaussianNB()
    }
    
    results = {}
    for name, model in traditional_models.items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_prob)
            # Get model params
            hyperparams = model.get_params() if hasattr(model, 'get_params') else {}
            results[name] = {
                'best_model': model,
                'best_score': val_auc,
                'best_params': hyperparams
            }
            print(f"  {name}: Val AUC = {val_auc:.4f}")
        except Exception as e:
            print(f"  {name}: Failed - {str(e)[:50]}")
    
    return results


def get_all_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_optuna_trials: int = 50, use_gpu: bool = True,
                            random_state: int = 42,
                            include_traditional: bool = True) -> Dict[str, Dict]:
    """
    Get all baseline models: both Optuna-optimized ensemble models and traditional models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_optuna_trials: Number of Optuna trials for ensemble models
        use_gpu: Whether to use GPU
        random_state: Random seed
        include_traditional: Whether to include traditional models (LR, SVM, etc.)
        
    Returns:
        Dictionary mapping model names to optimization results (with 'best_model', 'best_score', 'best_params')
    """
    all_results = {}
    
    # 1. Train Optuna-optimized ensemble models
    _, ensemble_results = get_optimized_boosting_models(
        X_train, y_train, X_val, y_val,
        n_trials=n_optuna_trials,
        use_gpu=use_gpu,
        random_state=random_state
    )
    all_results.update(ensemble_results)
    
    # 2. Train traditional baseline models
    if include_traditional:
        traditional_results = train_traditional_baselines(
            X_train, y_train, X_val, y_val, random_state=random_state
        )
        all_results.update(traditional_results)
    
    # Print summary
    print("\n" + "="*60)
    print("All Baseline Models Summary")
    print("="*60)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_score'], reverse=True)
    for name, result in sorted_results:
        print(f"  {name:<18} AUC = {result['best_score']:.4f}")
    print("="*60)
    best_name, best_result = sorted_results[0]
    print(f"[BEST] {best_name}: AUC = {best_result['best_score']:.4f}")
    
    return all_results


# =============================================================================
# Cross-Validation
# =============================================================================

def cross_validate_model(model_config: ModelConfig,
                         X: np.ndarray, y: np.ndarray,
                         n_folds: int = 5,
                         random_state: int = 42) -> Dict[str, Dict]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        model_config: ModelConfig object
        X: Feature matrix
        y: Labels
        n_folds: Number of folds
        random_state: Random seed
        
    Returns:
        Dictionary with mean and std of metrics
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone model
        model = model_config.model.__class__(**model_config.model.get_params())
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = evaluate_model(y_val, y_pred, y_proba)
        fold_metrics.append(metrics)
    
    # Aggregate results
    result = {}
    for metric in fold_metrics[0].keys():
        values = [m[metric] for m in fold_metrics]
        result[f'{metric}_mean'] = np.mean(values)
        result[f'{metric}_std'] = np.std(values)
    
    return result


# =============================================================================
# Model Selection
# =============================================================================

def select_best_model(results: Dict[str, Dict], 
                      metric: str = 'auc') -> Tuple[str, Dict]:
    """
    Select the best model based on a metric.
    
    Args:
        results: Dictionary of model results
        metric: Metric to use for selection
        
    Returns:
        Tuple of (best model name, best model results)
    """
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        raise ValueError("No valid model results")
    
    best_name = max(valid_results, key=lambda x: valid_results[x].get(metric, 0))
    return best_name, valid_results[best_name]


# =============================================================================
# Results Export
# =============================================================================

def results_to_dataframe(results: Dict[str, Dict]):
    """Convert results dictionary to pandas DataFrame."""
    import pandas as pd
    
    records = []
    for name, metrics in results.items():
        if 'error' not in metrics:
            record = {'model': name}
            record.update(metrics)
            records.append(record)
    
    return pd.DataFrame(records)


def generate_latex_table(results: Dict[str, Dict],
                         metrics: List[str] = None,
                         caption: str = 'Baseline model comparison',
                         label: str = 'tab:baseline') -> str:
    """
    Generate LaTeX table from results.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to include
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall']
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Model & " + " & ".join([m.upper() for m in metrics]) + " \\\\",
        "\\midrule"
    ]
    
    for name, result in results.items():
        if 'error' not in result:
            values = [f"{result.get(m, 0):.4f}" for m in metrics]
            lines.append(f"{name} & " + " & ".join(values) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)



