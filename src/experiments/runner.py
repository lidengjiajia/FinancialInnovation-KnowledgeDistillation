#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Runner for Credit Scoring
Handles baseline, distillation, and ablation experiments
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

from src.data.preprocessor import DataPreprocessor
from src.models.baselines import get_baseline_models, get_optimized_boosting_models
from src.models.neural import CreditNet, NeuralNetworkTrainer, device as neural_device
from src.distillation.losses import VanillaKDLoss, FitNetsLoss, AttentionLoss, SAKDLoss
from src.distillation.trainer import DistillationTrainer
from src.distillation.dt_distiller import DecisionTreeDistiller
from tqdm import tqdm


class ExperimentRunner:
    """Run all experiments for credit scoring."""
    
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc', 'brier']

    # Paper-consistent labels for tables/plots/exports
    METHOD_LABELS = {
        'Teacher': 'Teacher',
        'StudentBaseline': 'Student Baseline (DT)',
        'VanillaKD': 'VanillaKD',
        'SoftLabelKD': 'SoftLabelKD',
        # We use the adaptive variant as the default SHAP-KD in the paper/demo.
        'SHAPKD': 'SHAP-KD',
    }
    
    def __init__(self, data_dir: str = 'data', results_dir: str = 'results',
                 n_runs: int = 5, seed: int = 42, device_id: int = None,
                 force_retrain: bool = False,
                 generate_shap_plots: bool = True,
                 shap_trials: int = 30,
                 use_gpu_for_baseline: bool = True):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.n_runs = n_runs
        self.seed = seed
        self.device_id = device_id
        self.force_retrain = force_retrain
        self.generate_shap_plots = generate_shap_plots
        self.shap_trials = shap_trials
        self.use_gpu_for_baseline = use_gpu_for_baseline  # For CatBoost/XGBoost/LightGBM
        self.logger = logging.getLogger(__name__)
        
        # Set up device with explicit GPU ID if provided
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f'cuda:{device_id}')
            self.logger.info(f"ExperimentRunner using GPU-{device_id}: {torch.cuda.get_device_name(device_id)}")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        os.makedirs(results_dir, exist_ok=True)

    def _get_paper_figure_dir(self) -> str:
        """图片统一输出到 results/figures 目录"""
        fig_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        return fig_dir

    def _infer_teacher_name_from_distillation_results(self, dataset_name: str) -> Optional[str]:
        """Infer the teacher label used in the saved distillation results, if present.

        This keeps SHAP plots consistent with the teacher actually used in the experiments.
        Returns values like 'LightGBM-Tuned', 'XGBoost-Tuned', 'CatBoost-Tuned', or 'CreditNet'.
        """
        try:
            path = os.path.join(self.results_dir, f'{dataset_name}_distillation.xlsx')
            if not os.path.isfile(path):
                return None
            df = pd.read_excel(path)
            if 'teacher' not in df.columns:
                return None
            teachers = [t for t in df['teacher'].dropna().unique().tolist() if str(t).strip()]
            if not teachers:
                return None
            return str(teachers[0])
        except Exception:
            return None

    def run_shap_importance_plots(self, dataset_name: str) -> Optional[str]:
        """Generate SHAP feature-importance plot (Top-10) for one dataset.

        A方案（论文一致性）：使用该数据集蒸馏实验中实际采用的 Teacher 模型重新计算 SHAP，
        并覆盖保存到 Figure/shap_{dataset}_top10.png。
        """
        data = self._load_data(dataset_name)
        if data is None:
            return None

        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data.get('X_val'), data.get('y_val')
        X_test, y_test = data['X_test'], data['y_test']
        input_dim = X_train.shape[1]
        feature_names = data.get('feature_names', [f"Feature_{i}" for i in range(input_dim)])

        teacher_name = self._infer_teacher_name_from_distillation_results(dataset_name)
        teacher_model = None
        is_neural_teacher = False

        # 优先使用缓存的教师模型
        if hasattr(self, '_cached_teacher') and self._cached_teacher is not None:
            cached = self._cached_teacher
            teacher_model = cached['model']
            is_neural_teacher = cached['is_neural']
            teacher_name = cached['name']
            self.logger.info(f"    [SHAP Plot] Using cached teacher: {teacher_name} [cached]")
        # If distillation results exist, use that teacher label; otherwise fall back to runtime selection.
        elif teacher_name == 'CreditNet':
            # 尝试从磁盘加载缓存的CreditNet模型
            from src.experiments.teacher_trainer import check_teacher_cache
            cache = check_teacher_cache(dataset_name)
            if cache is not None and cache.best_model_name == 'CreditNet':
                teacher_model = cache.model
                is_neural_teacher = True
                self.logger.info(f"    [SHAP Plot] Loaded CreditNet from disk cache")
            else:
                train_loader, val_loader, _ = self._create_loaders(X_train, y_train, X_val, y_val, X_test, y_test)
                creditnet = CreditNet(input_dim, dataset_type=dataset_name).to(self.device)
                # SHAP plotting only: keep training bounded (early-stopping still applies)
                shap_epochs = 20 if dataset_name == 'uci' else 80
                self._train_creditnet_teacher(creditnet, train_loader, val_loader, dataset_name, epochs=shap_epochs)
                teacher_model = creditnet
                is_neural_teacher = True
                self.logger.info(f"    [SHAP Plot] Trained new CreditNet (no cache available)")
        elif teacher_name is not None:
            # 尝试从磁盘加载缓存的baseline模型
            from src.experiments.teacher_trainer import check_teacher_cache
            cache = check_teacher_cache(dataset_name)
            if cache is not None:
                # 尝试加载指定的baseline模型
                model_path = os.path.join(cache.cache_dir, f"{dataset_name}_teacher_{teacher_name}.pkl")
                if os.path.exists(model_path):
                    import pickle
                    with open(model_path, 'rb') as f:
                        teacher_model = pickle.load(f)
                    is_neural_teacher = False
                    self.logger.info(f"    [SHAP Plot] Loaded {teacher_name} from disk cache")
            if teacher_model is None:
                teacher_model = self._retrain_baseline_teacher(dataset_name, teacher_name, X_train, y_train)
                is_neural_teacher = False
                self.logger.info(f"    [SHAP Plot] Retrained {teacher_name} (no cache available)")
        else:
            teacher_model, is_neural_teacher = self._get_best_teacher(
                dataset_name, X_train, y_train, X_val, y_val, X_test, y_test
            )

        def _short_teacher_label(name: Optional[str], is_neural: bool) -> str:
            if name is None:
                return 'CreditNet' if is_neural else 'Auto'
            n = str(name)
            if 'LightGBM' in n:
                return 'LightGBM'
            if 'XGBoost' in n:
                return 'XGBoost'
            if 'CatBoost' in n:
                return 'CatBoost'
            if 'RF' in n or 'RandomForest' in n:
                return 'RF'
            if 'GBDT' in n:
                return 'GBDT'
            if 'CreditNet' in n:
                return 'CreditNet'
            return n.replace('-Tuned', '').strip()

        teacher_label = _short_teacher_label(teacher_name, is_neural_teacher)

        # --------------------------------------------------------------------
        # Compute SHAP values
        # --------------------------------------------------------------------
        try:
            import shap
        except Exception as e:
            self.logger.warning(f"  [SHAP] shap is required but not available: {e}")
            return None

        rng = np.random.default_rng(self.seed)
        bg_cap = 60 if is_neural_teacher else 120
        bg_n = min(bg_cap, len(X_train))
        eval_cap = 120 if is_neural_teacher else 500
        eval_n = min(eval_cap, len(X_test))
        bg_idx = rng.choice(len(X_train), size=bg_n, replace=False) if bg_n < len(X_train) else np.arange(len(X_train))
        X_bg = X_train[bg_idx]
        X_eval = X_test[:eval_n]

        shap_values = None

        if not is_neural_teacher:
            explainer = shap.TreeExplainer(teacher_model)
            shap_values = explainer.shap_values(X_eval)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            shap_values = np.asarray(shap_values)
        else:
            import torch

            def _predict_pos_proba(x_np: np.ndarray) -> np.ndarray:
                x_np = np.asarray(x_np, dtype=np.float32)
                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_np).to(self.device)
                    out = teacher_model(x_tensor).detach().cpu().numpy()

                out = np.asarray(out)
                # (n,2) logits -> softmax
                if out.ndim == 2 and out.shape[1] == 2:
                    out = out - out.max(axis=1, keepdims=True)
                    exp = np.exp(out)
                    proba = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
                    return proba[:, 1]

                out = out.squeeze()
                # If it looks like logits, apply sigmoid; otherwise assume already probability.
                if np.nanmin(out) < 0.0 or np.nanmax(out) > 1.0:
                    return 1.0 / (1.0 + np.exp(-out))
                return out

            # Prefer DeepExplainer when available (much faster for neural nets)
            try:
                teacher_model.eval()
                bg_t = torch.from_numpy(np.asarray(X_bg, dtype=np.float32)).to(self.device)
                eval_t = torch.from_numpy(np.asarray(X_eval, dtype=np.float32)).to(self.device)
                deep = shap.DeepExplainer(teacher_model, bg_t)
                shap_values = deep.shap_values(eval_t)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = np.asarray(shap_values)
            except Exception:
                explainer = shap.KernelExplainer(_predict_pos_proba, X_bg)
                shap_values = explainer.shap_values(X_eval, nsamples=80)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = np.asarray(shap_values)

        feature_importance = np.mean(np.abs(shap_values), axis=0).astype(float).reshape(-1)
        if len(feature_importance) != len(feature_names):
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        sorted_idx = np.argsort(-feature_importance)
        top_k = 10
        top_idx = sorted_idx[:top_k]
        top_features = [str(feature_names[i]) for i in top_idx]
        top_vals = [float(feature_importance[i]) for i in top_idx]

        # --------------------------------------------------------------------
        # Plot (paper-style horizontal bar) - 使用漂亮的渐变配色
        # --------------------------------------------------------------------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建高分辨率图表 - 增大尺寸以便更清晰
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        
        # 使用渐变色彩方案 - 从深到浅的蓝红渐变
        colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.8, len(top_features)))
        
        # 绘制水平条形图
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_vals, color=colors, edgecolor='#2C3E50', linewidth=1.2, alpha=0.85)
        
        # 在条形图上添加数值标签
        for i, (bar, val) in enumerate(zip(bars, top_vals)):
            ax.text(val + max(top_vals) * 0.01, i, f'{val:.4f}', 
                   va='center', fontsize=10, fontweight='bold', color='#2C3E50')
        
        # 设置标签 - 更大更清晰的字体
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11, fontweight='600')
        ax.invert_yaxis()  # 最重要的特征在顶部
        ax.set_xlabel('Mean |SHAP Value|', fontsize=13, fontweight='bold', color='#2C3E50')
        ax.set_title(
            f'SHAP Feature Importance - {dataset_name.upper()} (Teacher: {teacher_label})',
            fontsize=16,
            fontweight='bold',
            color='#1A252F',
            pad=20,
        )
        
        # 美化网格
        ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8, color='#7F8C8D')
        ax.set_axisbelow(True)
        
        # 设置背景色
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('white')
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor('#2C3E50')
            spine.set_linewidth(1.5)

        # 调整布局
        plt.tight_layout()

        fig_dir = self._get_paper_figure_dir()
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, f'shap_{dataset_name}_top10.png')
        
        # 保存高分辨率图片 (600 DPI)
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        plt.close(fig)

        self.logger.info(f"  [SHAP] Saved teacher-based feature-importance plot: {save_path}")
        return save_path
    
    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _safe_name(name: str) -> str:
        """Convert a display name into a filesystem-safe token."""
        return (
            name.replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('/', '-')
            .replace('\\', '-')
            .replace(':', '-')
        )
    
    def _load_data(self, dataset_name: str) -> Dict:
        preprocessor = DataPreprocessor(data_dir=self.data_dir, random_state=self.seed)
        loaders = {
            'german': preprocessor.load_german_credit,
            'australian': preprocessor.load_australian_credit,
            'xinwang': preprocessor.load_xinwang_credit,
            'uci': preprocessor.load_uci_credit
        }
        return loaders[dataset_name]()
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_prob: np.ndarray = None) -> Dict[str, float]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.5
        return metrics
    
    def _tune_threshold_and_compute_metrics(self, y_val: np.ndarray, val_prob: np.ndarray,
                                            y_test: np.ndarray, test_prob: np.ndarray) -> Dict[str, float]:
        """
        Tune decision threshold on validation set to maximize F1, then evaluate on test set.
        This ensures fair comparison between teacher and student models.
        
        Args:
            y_val: Validation labels
            val_prob: Validation probabilities
            y_test: Test labels
            test_prob: Test probabilities
        
        Returns:
            Dictionary of metrics computed with tuned threshold
        """
        # Grid search over thresholds on validation set
        grid = np.linspace(0.05, 0.95, 19)
        best_thr, best_f1 = 0.5, -1.0
        for thr in grid:
            y_pred_val = (val_prob >= thr).astype(int)
            f1 = f1_score(y_val, y_pred_val, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        
        # Apply tuned threshold to test set
        y_pred_test = (test_prob >= best_thr).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'threshold': best_thr
        }
        try:
            metrics['auc'] = roc_auc_score(y_test, test_prob)
            # PR-AUC: Area under Precision-Recall curve (better for imbalanced data)
            metrics['pr_auc'] = average_precision_score(y_test, test_prob)
            # Brier score: Calibration metric (lower is better)
            metrics['brier'] = brier_score_loss(y_test, test_prob)
        except:
            metrics['auc'] = 0.5
            metrics['pr_auc'] = 0.0
            metrics['brier'] = 1.0
        
        return metrics
    
    def _aggregate_results(self, all_runs: List[Dict]) -> Dict[str, str]:
        result = {}
        for metric in self.METRICS:
            values = [run[metric] for run in all_runs if metric in run]
            if values:
                mean, std = np.mean(values), np.std(values)
                result[metric] = f"{mean:.4f}±{std:.4f}"
                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
        return result
    
    # NOTE: model_cache functionality has been removed. 
    # All teacher model caching is now handled by TeacherTrainer in teacher_cache directory.
    
    def run_baseline_experiments(self, dataset_name: str, use_optuna: bool = True, 
                                  n_trials: int = 50, force_retrain: bool = False) -> pd.DataFrame:
        """
        Run baseline experiments with optional Optuna optimization for boosting models.
        
        Args:
            dataset_name: Name of dataset
            use_optuna: Whether to use Optuna for boosting models
            n_trials: Number of Optuna trials per model
            force_retrain: If True, ignore cached models and retrain
        """
        self.logger.info(f"  Running baseline experiments (Optuna: {use_optuna})...")
        baseline_path = os.path.join(self.results_dir, f'{dataset_name}_baseline.xlsx')

        # Fast path: reuse existing baseline sheet to avoid rerunning slow baselines (notably SVM).
        # Teacher selection for distillation can still benefit from tuned-model cache.
        if os.path.exists(baseline_path) and not force_retrain:
            if use_optuna:
                _, cached_results = self._load_optimized_models(dataset_name)
                if cached_results is not None:
                    self._optimization_results = cached_results
            try:
                df_cached = pd.read_excel(baseline_path)
                if 'model' in df_cached.columns and (df_cached['model'] == 'CreditNet').any():
                    self.logger.info(f"  [Cache] Reusing baseline sheet: {baseline_path}")
                    return df_cached
            except Exception as e:
                self.logger.warning(f"  [Cache] Failed to read baseline sheet; will recompute: {e}")

        data = self._load_data(dataset_name)
        if data is None:
            return None

        # Paper-consistent protocol: train/val/test (60/20/20) with z-score scaling
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.hstack([y_train, y_val])

        # Use GPU for baseline only if explicitly enabled (avoid CUDA conflicts in parallel mode)
        baseline_gpu = self.use_gpu_for_baseline and torch.cuda.is_available()
        models = get_baseline_models(use_gpu=baseline_gpu, random_state=self.seed)
        
        # Add Optuna-optimized boosting models if enabled
        if use_optuna:
            # Try to load cached models first
            if not force_retrain:
                cached_models, cached_results = self._load_optimized_models(dataset_name)
                if cached_models is not None:
                    # Remove non-tuned versions
                    ensemble_to_remove = ['RF', 'GBDT', 'XGBoost', 'LightGBM', 'CatBoost']
                    for model_name in ensemble_to_remove:
                        if model_name in models:
                            del models[model_name]
                    models.update(cached_models)
                    self._optimization_results = cached_results
                    self.logger.info(f"  [Cache] Using {len(cached_models)} cached tuned models")
                else:
                    cached_models = None
            else:
                cached_models = None
            
            # If no cache, run Optuna optimization
            if cached_models is None:
                self.logger.info(f"  [Optuna] Optimizing ensemble models ({n_trials} trials each)...")
                try:
                    X_train, y_train = data['X_train'], data['y_train']
                    X_val, y_val = data['X_val'], data['y_val']
                    
                    optimized_models, optimization_results = get_optimized_boosting_models(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials,
                        use_gpu=baseline_gpu,
                        random_state=self.seed
                    )
                    
                    # Remove non-tuned versions
                    ensemble_to_remove = ['RF', 'GBDT', 'XGBoost', 'LightGBM', 'CatBoost']
                    for model_name in ensemble_to_remove:
                        if model_name in models:
                            del models[model_name]
                    
                    models.update(optimized_models)
                    self._optimization_results = optimization_results
                    
                    # Save to cache
                    self._save_optimized_models(dataset_name, optimized_models, optimization_results)
                    
                    self.logger.info(f"  [Optuna] Replaced with {len(optimized_models)} tuned models")
                except Exception as e:
                    self.logger.warning(f"  [Optuna] Optimization failed: {e}")
                    self._optimization_results = {}
        
        results = []

        for name, config in models.items():
            self.logger.info(f"    {name}...")
            all_runs: List[Dict[str, float]] = []

            for run in range(self.n_runs):
                seed = self.seed + run
                self._set_seed(seed)

                # Clone model and inject run-specific random_state when supported
                model = config.model.__class__(**config.model.get_params())
                try:
                    params = model.get_params()
                    if 'random_state' in params:
                        model.set_params(random_state=seed)
                except Exception:
                    pass

                model.fit(X_trainval, y_trainval)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                all_runs.append(self._compute_metrics(y_test, y_pred, y_prob))

            agg = self._aggregate_results(all_runs)
            results.append({'model': name, 'category': config.category, **agg})
            if 'auc' in agg:
                self.logger.info(f"      AUC: {agg['auc']}")

        df = pd.DataFrame(results)
        baseline_path = os.path.join(self.results_dir, f'{dataset_name}_baseline.xlsx')
        df.to_excel(baseline_path, index=False)

        # Append/update CreditNet baseline (standard NN) into the same file
        df = self.add_neural_baseline_to_file(dataset_name)
        return df

    def run_neural_baseline_experiment(self, dataset_name: str, epochs: int = 100,
                                      batch_size: int | None = None) -> Dict[str, str]:
        """Run the standard CreditNet baseline on the fixed train/val/test split."""
        self.logger.info("  Running neural baseline (CreditNet) on train/val/test...")
        data = self._load_data(dataset_name)
        if data is None:
            return {}

        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']

        if batch_size is None:
            batch_size = 128 if dataset_name == 'uci' else 64
            if dataset_name == 'xinwang':
                batch_size = 64

        all_runs: List[Dict[str, float]] = []
        input_dim = X_train.shape[1]

        for run in range(self.n_runs):
            seed = self.seed + run
            self._set_seed(seed)

            train_loader, val_loader, test_loader = self._create_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
            )

            model = CreditNet(input_dim, dataset_type=dataset_name).to(self.device)
            self._train_creditnet_teacher(model, train_loader, val_loader, dataset_name, epochs=epochs)
            metrics = self._evaluate_creditnet(model, test_loader, dataset_name)
            all_runs.append(metrics)

        return self._aggregate_results(all_runs)

    def add_neural_baseline_to_file(self, dataset_name: str, epochs: int = 100,
                                    batch_size: int | None = None) -> pd.DataFrame:
        """Append/update the CreditNet baseline row into `<dataset>_baseline.xlsx`."""
        baseline_path = os.path.join(self.results_dir, f'{dataset_name}_baseline.xlsx')
        if os.path.exists(baseline_path):
            df = pd.read_excel(baseline_path)
        else:
            df = pd.DataFrame(columns=['model', 'category', *self.METRICS])

        nn_metrics = self.run_neural_baseline_experiment(dataset_name, epochs=epochs, batch_size=batch_size)
        if not nn_metrics:
            return df

        nn_row = {
            'model': 'CreditNet',
            'category': 'Neural',
            **nn_metrics
        }

        if (df['model'] == 'CreditNet').any():
            df.loc[df['model'] == 'CreditNet', nn_row.keys()] = list(nn_row.values())
        else:
            df = pd.concat([df, pd.DataFrame([nn_row])], ignore_index=True)

        df.to_excel(baseline_path, index=False)
        self.logger.info(f"  Updated baseline file with CreditNet: {baseline_path}")
        return df
    
    def run_distillation_experiments(self, dataset_name: str, use_best_teacher: bool = True) -> pd.DataFrame:
        """
        Run distillation experiments with Decision Tree as Student.
        
        Teacher: Automatically selected as the best performing model among 
                 Optuna-tuned baselines (RF, XGBoost, LightGBM, etc.) and CreditNet.
        Student: Decision Tree (sklearn DecisionTreeClassifier) - Fixed!
        
        This implements CB-KD framework:
        1. Select best Teacher (neural or ensemble)
        2. Distill knowledge to Decision Tree student via soft labels
        3. Apply class-balanced sample weighting
        4. Extract interpretable IF-THEN rules
        
        Args:
            dataset_name: Name of the dataset
            use_best_teacher: Whether to select the best model as Teacher
        """
        self.logger.info(f"  Running distillation experiments (Student=DecisionTree)...")
        data = self._load_data(dataset_name)
        if data is None:
            return None
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        input_dim = X_train.shape[1]
        
        # Get feature names if available
        feature_names = data.get('feature_names', [f"Feature_{i}" for i in range(input_dim)])
        
        # ====================================================================
        # Step 1: Find the best Teacher model (use cache if available)
        # ====================================================================
        best_teacher_info = None
        best_teacher_auc = 0.0
        best_teacher_model = None
        is_neural_teacher = False
        
        # 首先检查是否有缓存的教师模型
        if hasattr(self, '_cached_teacher') and self._cached_teacher is not None:
            cached = self._cached_teacher
            best_teacher_model = cached['model']
            is_neural_teacher = cached['is_neural']
            best_teacher_auc = cached['auc']
            best_teacher_info = {
                'type': 'neural' if is_neural_teacher else 'baseline',
                'name': cached['name'],
                'auc': cached['auc'],
                'model': cached['model']
            }
            self.logger.info(f"    [Best Teacher] {cached['name']} (AUC: {best_teacher_auc:.4f}) [cached]")
        elif use_best_teacher:
            self.logger.info(f"    [Teacher Selection] Comparing baseline and neural models...")
            
            # Get optimization results from baseline experiments (if available)
            if hasattr(self, '_optimization_results') and self._optimization_results:
                for model_name, result in self._optimization_results.items():
                    if result['best_score'] > best_teacher_auc:
                        best_teacher_auc = result['best_score']
                        best_teacher_info = {
                            'type': 'baseline',
                            'name': model_name,
                            'auc': result['best_score'],
                            'model': result.get('best_model', None)
                        }
            
            # Train and evaluate CreditNet on validation to compare (no test leakage)
            train_loader, val_loader, test_loader = self._create_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            creditnet = CreditNet(input_dim, dataset_type=dataset_name).to(self.device)
            self._train_creditnet_teacher(creditnet, train_loader, val_loader, dataset_name)
            creditnet_metrics = self._evaluate_creditnet(creditnet, val_loader, dataset_name)
            creditnet_auc = creditnet_metrics.get('auc', 0.0)
            
            self.logger.info(f"      CreditNet AUC: {creditnet_auc:.4f}")
            
            if creditnet_auc > best_teacher_auc:
                best_teacher_auc = creditnet_auc
                best_teacher_info = {
                    'type': 'neural',
                    'name': 'CreditNet',
                    'auc': creditnet_auc
                }
                best_teacher_model = creditnet
                is_neural_teacher = True
            else:
                # Use the best baseline model
                if best_teacher_info and best_teacher_info.get('model') is not None:
                    best_teacher_model = best_teacher_info['model']
                    is_neural_teacher = False
                else:
                    # Fallback: retrain the best baseline
                    best_teacher_model = self._retrain_baseline_teacher(
                        dataset_name, best_teacher_info['name'], X_train, y_train
                    )
                    is_neural_teacher = False
            
            if best_teacher_info:
                self.logger.info(f"    [Best Teacher] {best_teacher_info['name']} (AUC: {best_teacher_info['auc']:.4f})")
            else:
                self.logger.warning(f"    [Warning] Could not determine best teacher, using CreditNet")
                best_teacher_model = creditnet
                is_neural_teacher = True
                best_teacher_info = {'type': 'neural', 'name': 'CreditNet', 'auc': creditnet_auc}
        
        # ====================================================================
        # Step 2: Define distillation methods (all with DT student)
        # ====================================================================
        dt_configs = {
            # Baselines
            'Teacher': {'skip_distillation': True, 'use_teacher': True},
            'StudentBaseline': {'skip_distillation': True, 'use_teacher': False},
            # Standard KD methods
            # VanillaKD: Pure soft-label distillation without class balance
            'VanillaKD': {'temperature': 4.0, 'alpha': 0.0, 'max_depth': 6,
                         'use_class_balance': False},
            # SoftLabelKD: Soft-label KD with hard-label weight (no class balance)
            'SoftLabelKD': {'temperature': 4.0, 'alpha': 0.2, 'max_depth': 6,
                           'use_class_balance': False},
            # CB-KD: Our method - same as SoftLabelKD but WITH class-balanced weighting
            # This makes the comparison fair: the only difference is class balance
            'CBKD': {'temperature': 4.0, 'alpha': 0.2, 'max_depth': 6,
                     'use_class_balance': True},
        }
        
        results = []
        extracted_rules = {}
        
        # ====================================================================
        # Step 3: Run distillation experiments for each method
        # ====================================================================
        for method_name, config in dt_configs.items():
            self.logger.info(f"    {method_name}...")
            all_runs = []
            
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                
                if config.get('skip_distillation') and config.get('use_teacher'):
                    # Evaluate the teacher with validation-tuned threshold (fair comparison)
                    if is_neural_teacher:
                        train_loader, val_loader, test_loader = self._create_loaders(
                            X_train, y_train, X_val, y_val, X_test, y_test
                        )
                        # For neural teacher, use tuned threshold evaluation
                        import torch
                        best_teacher_model.eval()
                        with torch.no_grad():
                            # Get validation probabilities
                            val_probs_list = []
                            for X_batch, _ in val_loader:
                                out = best_teacher_model(X_batch.to(self.device))
                                if out.shape[1] == 2:
                                    probs = torch.softmax(out, dim=1)[:, 1]
                                else:
                                    probs = torch.sigmoid(out).squeeze()
                                val_probs_list.append(probs.cpu().numpy())
                            val_prob = np.concatenate(val_probs_list)
                            
                            # Get test probabilities
                            test_probs_list = []
                            for X_batch, _ in test_loader:
                                out = best_teacher_model(X_batch.to(self.device))
                                if out.shape[1] == 2:
                                    probs = torch.softmax(out, dim=1)[:, 1]
                                else:
                                    probs = torch.sigmoid(out).squeeze()
                                test_probs_list.append(probs.cpu().numpy())
                            test_prob = np.concatenate(test_probs_list)
                        
                        metrics = self._tune_threshold_and_compute_metrics(
                            y_val, val_prob, y_test, test_prob
                        )
                    else:
                        # For ensemble teacher, use tuned threshold evaluation
                        val_prob = best_teacher_model.predict_proba(X_val)[:, 1]
                        test_prob = best_teacher_model.predict_proba(X_test)[:, 1]
                        metrics = self._tune_threshold_and_compute_metrics(
                            y_val, val_prob, y_test, test_prob
                        )
                
                elif config.get('skip_distillation') and not config.get('use_teacher'):
                    # Train DT without distillation (baseline) with tuned threshold
                    from sklearn.tree import DecisionTreeClassifier
                    dt = DecisionTreeClassifier(
                        max_depth=6, min_samples_leaf=5, 
                        random_state=self.seed + run, class_weight='balanced'
                    )
                    dt.fit(X_train, y_train)
                    val_prob = dt.predict_proba(X_val)[:, 1]
                    test_prob = dt.predict_proba(X_test)[:, 1]
                    metrics = self._tune_threshold_and_compute_metrics(
                        y_val, val_prob, y_test, test_prob
                    )
                
                else:
                    # Knowledge distillation to Decision Tree
                    distiller = DecisionTreeDistiller(
                        temperature=config.get('temperature', 4.0),
                        alpha=config.get('alpha', 0.0),
                        max_depth=config.get('max_depth', 6),
                        min_samples_leaf=5,
                        use_class_balance=config.get('use_class_balance', True),
                        random_state=self.seed + run
                    )
                    
                    distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                    distiller.fit(X_train, y_train, X_val, y_val, 
                                 feature_names=feature_names)
                    
                    metrics = distiller.evaluate(X_test, y_test)
                    metrics['kt_score'] = distiller.knowledge_transfer_score
                    
                    # Extract rules on the last run for CB-KD
                    if run == self.n_runs - 1 and method_name == 'CBKD':
                        rules = distiller.extract_rules(feature_names)
                        extracted_rules[method_name] = {
                            'rules': rules,
                            'text': distiller.rules_to_text(rules),
                            'tree_depth': distiller.student.get_depth(),
                            'n_leaves': distiller.student.get_n_leaves(),
                            'kt_score': distiller.knowledge_transfer_score
                        }
                
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            results.append({
                'method': self.METHOD_LABELS.get(method_name, method_name),
                'is_ours': method_name.startswith('SHAPKD'),
                'teacher': best_teacher_info['name'] if best_teacher_info else 'CreditNet',
                **agg
            })
            self.logger.info(f"      AUC: {agg['auc']}")
        
        # ====================================================================
        # Step 4: Save results and extracted rules
        # ====================================================================
        df = pd.DataFrame(results)
        df.to_excel(os.path.join(self.results_dir, f'{dataset_name}_distillation.xlsx'), index=False)
        
        # Save extracted rules
        if extracted_rules:
            rules_dir = os.path.join(self.results_dir, 'rules')
            os.makedirs(rules_dir, exist_ok=True)
            
            for method_name, rule_data in extracted_rules.items():
                method_label = self.METHOD_LABELS.get(method_name, method_name)
                # Save as text file
                rule_file = os.path.join(rules_dir, f'{dataset_name}_{self._safe_name(method_label)}_rules.txt')
                with open(rule_file, 'w', encoding='utf-8') as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Method: {method_label}\n")
                    f.write(f"Teacher: {best_teacher_info['name'] if best_teacher_info else 'CreditNet'}\n")
                    f.write(f"Tree Depth: {rule_data['tree_depth']}\n")
                    f.write(f"Number of Leaves: {rule_data['n_leaves']}\n")
                    f.write("\n")
                    f.write(rule_data['text'])
                
                self.logger.info(f"    [Rules] Saved to {rule_file}")
                
                # Save rules as DataFrame
                rule_df = pd.DataFrame([
                    {
                        'Rule_ID': r['rule_id'],
                        'Conditions': ' AND '.join([f"{f} {o} {t:.4f}" for f, o, t in r['conditions']]),
                        'Prediction': r['prediction'],
                        'Samples': r['samples'],
                        'Confidence': r['confidence']
                    }
                    for r in rule_data['rules']
                ])
                rule_df.to_excel(
                    os.path.join(rules_dir, f'{dataset_name}_{self._safe_name(method_label)}_rules.xlsx'),
                    index=False
                )
        
        return df
    
    def _retrain_baseline_teacher(self, dataset_name: str, model_name: str, 
                                   X_train: np.ndarray, y_train: np.ndarray):
        """Retrain a baseline model to use as teacher."""
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        model_map = {
            'RF-Tuned': RandomForestClassifier(n_estimators=200, random_state=self.seed),
            'XGBoost-Tuned': XGBClassifier(n_estimators=200, random_state=self.seed, verbosity=0),
            'LightGBM-Tuned': LGBMClassifier(n_estimators=200, random_state=self.seed, verbose=-1),
            'CatBoost-Tuned': CatBoostClassifier(iterations=200, random_state=self.seed, verbose=0),
            'GBDT-Tuned': GradientBoostingClassifier(n_estimators=200, random_state=self.seed)
        }
        
        if model_name in model_map:
            model = model_map[model_name]
            model.fit(X_train, y_train)
            return model
        
        # Default to XGBoost
        model = XGBClassifier(n_estimators=200, random_state=self.seed, verbosity=0)
        model.fit(X_train, y_train)
        return model
    
    def run_ablation_experiments(self, dataset_name: str) -> pd.DataFrame:
        """
        Run comprehensive ablation experiments with Decision Tree as Student.
        
        Ablation dimensions:
        1. Temperature: Controls soft label smoothness
        2. Alpha: Balance between soft and hard labels
        3. Max Depth: Decision tree complexity
        4. Class Balance: Effect of class-balanced sample weighting
        """
        self.logger.info(f"  Running ablation experiments (Student=DecisionTree)...")
        data = self._load_data(dataset_name)
        if data is None:
            return None
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data.get('feature_names', [f"Feature_{i}" for i in range(X_train.shape[1])])
        
        # Get best teacher from baseline experiments
        best_teacher_model, is_neural_teacher = self._get_best_teacher(
            dataset_name, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        all_results = []
        
        # ====================================================================
        # Ablation 1: Temperature Scaling
        # ====================================================================
        self.logger.info(f"    Ablation 1: Temperature (\u03c4)")
        temperatures = [1.0, 2.0, 4.0, 8.0]
        
        for value in temperatures:
            all_runs = []
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                distiller = DecisionTreeDistiller(
                    temperature=value, alpha=0.0, max_depth=6,
                    min_samples_leaf=5, use_class_balance=True,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, 
                             feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['tree_depth'] = distiller.student.get_depth()
                metrics['n_leaves'] = distiller.student.get_n_leaves()
                metrics['kt_score'] = distiller.knowledge_transfer_score
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_kt = np.mean([r['kt_score'] for r in all_runs])
            all_results.append({
                'ablation_type': 'temperature',
                'value': value,
                'avg_tree_depth': np.mean([r['tree_depth'] for r in all_runs]),
                'avg_n_leaves': np.mean([r['n_leaves'] for r in all_runs]),
                'avg_kt_score': avg_kt,
                **agg
            })
            self.logger.info(f"      T={value}: AUC={agg['auc']}, KT={avg_kt:.4f}")
        
        # ====================================================================
        # Ablation 2: Alpha (Soft vs Hard Label Balance)
        # ====================================================================
        self.logger.info(f"    Ablation 2: Alpha (hard-label weight)")
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        for value in alphas:
            all_runs = []
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=value, max_depth=6,
                    min_samples_leaf=5, use_class_balance=True,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, 
                             feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['tree_depth'] = distiller.student.get_depth()
                metrics['kt_score'] = distiller.knowledge_transfer_score
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_kt = np.mean([r['kt_score'] for r in all_runs])
            label_type = 'SoftOnly' if value == 0.0 else ('HardOnly' if value == 1.0 else 'Mixed')
            all_results.append({
                'ablation_type': 'alpha',
                'value': value,
                'label_type': label_type,
                'avg_kt_score': avg_kt,
                **agg
            })
            self.logger.info(f"      alpha={value} ({label_type}): AUC={agg['auc']}")
        
        # ====================================================================
        # Ablation 3: Max Depth (Tree Complexity)
        # ====================================================================
        self.logger.info(f"    Ablation 3: Max Depth")
        depths = [3, 4, 6, 8]
        
        for value in depths:
            all_runs = []
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=0.0, max_depth=value,
                    min_samples_leaf=5, use_class_balance=True,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, 
                             feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['tree_depth'] = distiller.student.get_depth()
                metrics['n_leaves'] = distiller.student.get_n_leaves()
                metrics['kt_score'] = distiller.knowledge_transfer_score
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_depth = np.mean([r['tree_depth'] for r in all_runs])
            avg_leaves = np.mean([r['n_leaves'] for r in all_runs])
            all_results.append({
                'ablation_type': 'max_depth',
                'value': value,
                'actual_depth': avg_depth,
                'avg_n_leaves': avg_leaves,
                'n_rules': avg_leaves,
                **agg
            })
            self.logger.info(f"      max_depth={value}: AUC={agg['auc']}, leaves={avg_leaves:.0f}")
        
        # ====================================================================
        # Ablation 4: Class Balance vs No Class Balance
        # ====================================================================
        self.logger.info(f"    Ablation 4: Class Balance Effect")
        class_balance_options = [False, True]

        for use_cb in class_balance_options:
            all_runs = []
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=0.0, max_depth=6,
                    min_samples_leaf=5,
                    use_class_balance=use_cb,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, 
                             feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['kt_score'] = distiller.knowledge_transfer_score
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_kt = np.mean([r['kt_score'] for r in all_runs])
            all_results.append({
                'ablation_type': 'class_balance',
                'value': 1 if use_cb else 0,
                'label': 'CB-KD' if use_cb else 'SoftLabelKD',
                'avg_kt_score': avg_kt,
                **agg
            })
            self.logger.info(f"      class_balance={use_cb}: AUC={agg['auc']}, KT={avg_kt:.4f}")
        
        # ====================================================================
        # Ablation 5: Split Criterion (Gini vs Entropy vs Log_loss)
        # ====================================================================
        self.logger.info(f"    Ablation 5: Split Criterion")
        criteria = ['gini', 'entropy', 'log_loss']
        
        for criterion in criteria:
            all_runs = []
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=0.0, max_depth=6,
                    min_samples_leaf=5, use_class_balance=True,
                    criterion=criterion,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, 
                             feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['kt_score'] = distiller.knowledge_transfer_score
                metrics['n_leaves'] = distiller.student.get_n_leaves()
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_kt = np.mean([r['kt_score'] for r in all_runs])
            avg_leaves = np.mean([r['n_leaves'] for r in all_runs])
            all_results.append({
                'ablation_type': 'criterion',
                'value': criterion,
                'avg_kt_score': avg_kt,
                'avg_n_leaves': avg_leaves,
                **agg
            })
            self.logger.info(f"      criterion={criterion}: AUC={agg['auc']}, leaves={avg_leaves:.0f}")
        
        # ====================================================================
        # Ablation 6: Cost Ratio (Cost-Sensitive Learning)
        # ====================================================================
        self.logger.info(f"    Ablation 6: Cost Ratio (minority class weight multiplier)")
        cost_ratios = [1.0, 2.0, 3.0, 5.0, 8.0]
        
        for cost_ratio in cost_ratios:
            all_runs = []
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=0.0, max_depth=6,
                    min_samples_leaf=5, use_class_balance=True,
                    cost_ratio=cost_ratio,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, 
                             feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['kt_score'] = distiller.knowledge_transfer_score
                metrics['n_leaves'] = distiller.student.get_n_leaves()
                
                # Count default rules
                rules = distiller.extract_rules(feature_names)
                default_rules = [r for r in rules if r['prediction_code'] == 1]
                metrics['n_default_rules'] = len(default_rules)
                metrics['n_total_rules'] = len(rules)
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_kt = np.mean([r['kt_score'] for r in all_runs])
            avg_default_rules = np.mean([r['n_default_rules'] for r in all_runs])
            avg_total_rules = np.mean([r['n_total_rules'] for r in all_runs])
            all_results.append({
                'ablation_type': 'cost_ratio',
                'value': cost_ratio,
                'avg_kt_score': avg_kt,
                'avg_default_rules': avg_default_rules,
                'avg_total_rules': avg_total_rules,
                **agg
            })
            self.logger.info(f"      cost_ratio={cost_ratio}: AUC={agg['auc']}, default_rules={avg_default_rules:.1f}/{avg_total_rules:.0f}")
        
        df = pd.DataFrame(all_results)
        df.to_excel(os.path.join(self.results_dir, f'{dataset_name}_ablation.xlsx'), index=False)
        return df
    
    def _get_best_teacher(self, dataset_name: str, X_train, y_train, X_val, y_val, X_test, y_test):
        """Get the best teacher model (cached if available)."""
        input_dim = X_train.shape[1]
        
        # 首先检查是否有缓存的教师模型
        if hasattr(self, '_cached_teacher') and self._cached_teacher is not None:
            cached = self._cached_teacher
            self.logger.info(f"    [Best Teacher] {cached['name']} (AUC: {cached['auc']:.4f}) [cached]")
            return cached['model'], cached['is_neural']
        
        self.logger.info(f"    [Teacher Selection] Comparing baseline and neural models...")
        
        best_teacher_model = None
        best_teacher_auc = 0.0
        best_teacher_name = None
        is_neural_teacher = False
        
        # Check baseline optimization results
        if hasattr(self, '_optimization_results') and self._optimization_results:
            for model_name, result in self._optimization_results.items():
                if result['best_score'] > best_teacher_auc:
                    best_teacher_auc = result['best_score']
                    best_teacher_model = result.get('best_model', None)
                    best_teacher_name = model_name
                    is_neural_teacher = False
        
        # Compare with CreditNet on validation (no test leakage)
        train_loader, val_loader, test_loader = self._create_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        creditnet = CreditNet(input_dim, dataset_type=dataset_name).to(self.device)
        self._train_creditnet_teacher(creditnet, train_loader, val_loader, dataset_name)
        creditnet_metrics = self._evaluate_creditnet(creditnet, val_loader, dataset_name)
        creditnet_auc = creditnet_metrics.get('auc', 0.0)
        self.logger.info(f"      CreditNet AUC: {creditnet_auc:.4f}")
        
        if creditnet_auc > best_teacher_auc:
            best_teacher_model = creditnet
            best_teacher_name = 'CreditNet'
            best_teacher_auc = creditnet_auc
            is_neural_teacher = True
        elif best_teacher_model is None:
            best_teacher_model = creditnet
            best_teacher_name = 'CreditNet'
            best_teacher_auc = creditnet_auc
            is_neural_teacher = True
        
        self.logger.info(f"    [Best Teacher] {best_teacher_name} (AUC: {best_teacher_auc:.4f})")
        
        # Cache the teacher for later use
        self._cached_teacher = {
            'model': best_teacher_model,
            'name': best_teacher_name,
            'auc': best_teacher_auc,
            'is_neural': is_neural_teacher
        }
        
        return best_teacher_model, is_neural_teacher
    
    def run_theoretical_analysis(self, dataset_name: str) -> pd.DataFrame:
        """
        Run comprehensive theoretical analysis experiments with Decision Tree as Student.
        
        Validates knowledge distillation theory and our innovations:
        1. Temperature scaling effect: Higher T → smoother soft labels → better generalization
        2. Soft vs Hard labels: Optimal alpha balances teacher knowledge and ground truth
        3. Tree complexity: Deeper trees capture more knowledge but risk overfitting
        4. Knowledge transfer quality: Measures how well student mimics teacher
        5. SHAP contribution analysis: Validates SHAP-guided weighting effectiveness
        """
        self.logger.info(f"  Running theoretical analysis (Student=DecisionTree)...")
        data = self._load_data(dataset_name)
        if data is None:
            return None
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data.get('feature_names', [f"Feature_{i}" for i in range(X_train.shape[1])])
        
        # Get best teacher
        best_teacher_model, is_neural_teacher = self._get_best_teacher(
            dataset_name, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        results = []
        
        # ====================================================================
        # Analysis 1: Temperature Scaling Effect
        # Theoretical basis: Higher T produces softer probability distributions,
        # revealing "dark knowledge" about class similarities
        # ====================================================================
        self.logger.info("    Analysis 1: Temperature scaling effect")
        temperatures = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        
        for temp in temperatures:
            all_runs = []
            
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                
                distiller = DecisionTreeDistiller(
                    temperature=temp, alpha=0.0, max_depth=6,
                    min_samples_leaf=5, use_class_balance=True,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                
                # Collect additional analysis metrics
                metrics['tree_depth'] = distiller.student.get_depth()
                metrics['n_leaves'] = distiller.student.get_n_leaves()
                metrics['kt_score'] = distiller.knowledge_transfer_score
                metrics['agreement'] = distiller.kt_metrics['agreement_rate']
                metrics['prob_corr'] = distiller.kt_metrics['probability_correlation']
                metrics['rank_corr'] = distiller.kt_metrics['rank_correlation']
                
                # Soft label entropy (measure of label smoothness)
                soft_probs = distiller.soft_labels
                entropy = -np.mean(soft_probs * np.log(soft_probs + 1e-8) + 
                                   (1-soft_probs) * np.log(1-soft_probs + 1e-8))
                metrics['soft_label_entropy'] = entropy
                
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            results.append({
                'experiment': 'TemperatureScaling',
                'parameter': 'temperature',
                'value': temp,
                'avg_depth': np.mean([r['tree_depth'] for r in all_runs]),
                'avg_kt_score': np.mean([r['kt_score'] for r in all_runs]),
                'avg_agreement': np.mean([r['agreement'] for r in all_runs]),
                'avg_entropy': np.mean([r['soft_label_entropy'] for r in all_runs]),
                **agg
            })
            self.logger.info(f"      T={temp}: AUC={agg['auc']}, KT={np.mean([r['kt_score'] for r in all_runs]):.4f}")
        
        # ====================================================================
        # Analysis 2: Soft vs Hard Label Balance (Knowledge-Reality Trade-off)
        # Theoretical basis: α controls trade-off between:
        # - Teacher's "dark knowledge" (soft labels)
        # - Ground truth supervision (hard labels)
        # ====================================================================
        self.logger.info("    Analysis 2: Soft vs Hard label balance")
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for alpha in alphas:
            all_runs = []
            
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=alpha, max_depth=6,
                    min_samples_leaf=5, use_class_balance=True,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['tree_depth'] = distiller.student.get_depth()
                metrics['kt_score'] = distiller.knowledge_transfer_score
                metrics['agreement'] = distiller.kt_metrics['agreement_rate']
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            label_type = 'SoftOnly' if alpha == 0.0 else ('HardOnly' if alpha == 1.0 else 'Mixed')
            results.append({
                'experiment': 'LabelBalance',
                'parameter': 'alpha',
                'value': alpha,
                'label_type': label_type,
                'avg_kt_score': np.mean([r['kt_score'] for r in all_runs]),
                'avg_agreement': np.mean([r['agreement'] for r in all_runs]),
                **agg
            })
            self.logger.info(f"      alpha={alpha} ({label_type}): AUC={agg['auc']}")
        
        # ====================================================================
        # Analysis 3: Tree Complexity vs Performance (Interpretability Trade-off)
        # Theoretical basis: Deeper trees can capture more complex patterns but:
        # - Reduced interpretability (more rules)
        # - Risk of overfitting
        # ====================================================================
        self.logger.info("    Analysis 3: Tree complexity analysis")
        depths = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
        
        for depth in depths:
            all_runs = []
            
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                
                distiller = DecisionTreeDistiller(
                    temperature=4.0, alpha=0.0, max_depth=depth,
                    min_samples_leaf=5, use_class_balance=True,
                    random_state=self.seed + run
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['actual_depth'] = distiller.student.get_depth()
                metrics['n_leaves'] = distiller.student.get_n_leaves()
                metrics['n_rules'] = distiller.student.get_n_leaves()
                metrics['kt_score'] = distiller.knowledge_transfer_score
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            avg_actual_depth = np.mean([r['actual_depth'] for r in all_runs])
            avg_rules = np.mean([r['n_rules'] for r in all_runs])
            avg_kt = np.mean([r['kt_score'] for r in all_runs])
            
            results.append({
                'experiment': 'TreeComplexity',
                'parameter': 'max_depth',
                'value': depth,
                'actual_depth': avg_actual_depth,
                'n_rules': avg_rules,
                'avg_kt_score': avg_kt,
                **agg
            })
            self.logger.info(f"      max_depth={depth}: AUC={agg['auc']}, rules={avg_rules:.0f}")
        
        # ====================================================================
        # Analysis 4: Knowledge Transfer Quality
        # Validates that higher KT scores correlate with better student performance
        # ====================================================================
        self.logger.info("    Analysis 4: Knowledge transfer quality")
        kt_analysis_configs = [
            ('VanillaKD', {'use_class_balance': False, 'alpha': 1.0}),
            ('SoftLabelKD', {'use_class_balance': False, 'alpha': 0.0}),
            ('CB-KD', {'use_class_balance': True, 'alpha': 0.0}),
        ]
        
        for config_name, config in kt_analysis_configs:
            all_runs = []
            
            for run in range(self.n_runs):
                self._set_seed(self.seed + run)
                
                distiller = DecisionTreeDistiller(
                    temperature=4.0, max_depth=6, min_samples_leaf=5,
                    random_state=self.seed + run, **config
                )
                distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
                distiller.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
                metrics = distiller.evaluate(X_test, y_test)
                metrics['kt_score'] = distiller.knowledge_transfer_score
                metrics['agreement'] = distiller.kt_metrics['agreement_rate']
                metrics['prob_corr'] = distiller.kt_metrics['probability_correlation']
                metrics['rank_corr'] = distiller.kt_metrics['rank_correlation']
                all_runs.append(metrics)
            
            agg = self._aggregate_results(all_runs)
            results.append({
                'experiment': 'KnowledgeTransfer',
                'parameter': 'config',
                'value': config_name,
                'avg_kt_score': np.mean([r['kt_score'] for r in all_runs]),
                'avg_agreement': np.mean([r['agreement'] for r in all_runs]),
                'avg_prob_corr': np.mean([r['prob_corr'] for r in all_runs]),
                'avg_rank_corr': np.mean([r['rank_corr'] for r in all_runs]),
                **agg
            })
            self.logger.info(f"      {config_name}: AUC={agg['auc']}, KT={np.mean([r['kt_score'] for r in all_runs]):.4f}")
        
        # ====================================================================
        # Analysis 5: Class Balance Effect
        # Compares CB-KD vs SoftLabelKD
        # ====================================================================
        self.logger.info("    Analysis 5: Class balance effect")
        
        # Run with CB-KD
        self._set_seed(self.seed)
        distiller = DecisionTreeDistiller(
            temperature=4.0, alpha=0.0, max_depth=6,
            min_samples_leaf=5, use_class_balance=True,
            random_state=self.seed
        )
        distiller.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
        distiller.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
        
        # Report class balance effect
        cb_metrics = distiller.evaluate(X_test, y_test)
        results.append({
            'experiment': 'ClassBalance',
            'parameter': 'use_class_balance',
            'value': 'CB-KD',
            'kt_score': distiller.knowledge_transfer_score,
            'auc': cb_metrics['auc'],
            'f1': cb_metrics['f1'],
        })
        self.logger.info(f"      CB-KD: AUC={cb_metrics['auc']:.4f}, F1={cb_metrics['f1']:.4f}")
        
        # Compare with no class balance
        self._set_seed(self.seed)
        distiller_nocb = DecisionTreeDistiller(
            temperature=4.0, alpha=0.0, max_depth=6,
            min_samples_leaf=5, use_class_balance=False,
            random_state=self.seed
        )
        distiller_nocb.set_teacher(best_teacher_model, is_neural=is_neural_teacher)
        distiller_nocb.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
        nocb_metrics = distiller_nocb.evaluate(X_test, y_test)
        results.append({
            'experiment': 'ClassBalance',
            'parameter': 'use_class_balance',
            'value': 'SoftLabelKD',
            'kt_score': distiller_nocb.knowledge_transfer_score,
            'auc': nocb_metrics['auc'],
            'f1': nocb_metrics['f1'],
        })
        self.logger.info(f"      SoftLabelKD: AUC={nocb_metrics['auc']:.4f}, F1={nocb_metrics['f1']:.4f}")
        
        df = pd.DataFrame(results)
        df.to_excel(os.path.join(self.results_dir, f'{dataset_name}_theoretical_analysis.xlsx'), index=False)
        return df
    
    def run_all_experiments(self, dataset_name: str, include_theory: bool = True) -> Dict[str, pd.DataFrame]:
        # Print device info at experiment start
        device_info = f"{self.device}"
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device.index if self.device.index else 0)
            device_info = f"{self.device} ({gpu_name})"
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Running all experiments for {dataset_name.upper()}")
        self.logger.info(f"Device: {device_info}")
        self.logger.info(f"Force retrain: {self.force_retrain}")
        self.logger.info(f"{'='*50}")
        
        results = {}
        
        self.logger.info(f"  [1/4] Baseline experiments...")
        results['baseline'] = self.run_baseline_experiments(dataset_name, force_retrain=self.force_retrain)

        if self.generate_shap_plots:
            self.logger.info(f"  [SHAP] Generating SHAP importance plot...")
            self.run_shap_importance_plots(dataset_name)
        
        self.logger.info(f"  [2/4] Distillation experiments...")
        results['distillation'] = self.run_distillation_experiments(dataset_name)
        
        self.logger.info(f"  [3/4] Ablation experiments...")
        results['ablation'] = self.run_ablation_experiments(dataset_name)
        
        if include_theory:
            self.logger.info(f"  [4/4] Theoretical analysis...")
            results['theoretical_analysis'] = self.run_theoretical_analysis(dataset_name)
        
        self.logger.info(f"All experiments completed for {dataset_name.upper()} on {device_info}")
        return results
    
    def _compute_shap_values(self, X: np.ndarray, y: np.ndarray):
        try:
            import shap
            from xgboost import XGBClassifier
            
            xgb = XGBClassifier(n_estimators=50, max_depth=4, random_state=self.seed, 
                               verbosity=0, use_label_encoder=False)
            xgb.fit(X, y)
            
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X[:min(500, len(X))])
            return shap_values
        except Exception as e:
            self.logger.warning(f"SHAP computation failed: {e}")
            return None
    
    def _create_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size)
        )
    
    def _train_model(self, model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, epochs: int = 100):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        patience = 15
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    val_loss += criterion(model(X), y).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        model.eval()
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                logits = model(X)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return self._compute_metrics(
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs)
        )
    
    def _train_creditnet_teacher(self, model: nn.Module, train_loader: DataLoader,
                                  val_loader: DataLoader, dataset_name: str, epochs: int = 150):
        """Train CreditNet teacher model with dataset-specific optimization.
        
        All datasets now use logits mode (BCEWithLogitsLoss) for consistency
        and better numerical stability during knowledge distillation.
        """
        import torch.optim as optim
        
        # Dataset-specific hyperparameters
        if dataset_name == 'uci':
            learning_rate = 0.001
            weight_decay = 1e-4
            patience = 20
        elif dataset_name == 'australian':
            learning_rate = 0.002
            weight_decay = 1e-3
            patience = 15
        elif dataset_name == 'xinwang':
            learning_rate = 0.001
            weight_decay = 1e-3
            patience = 20
        else:  # german
            learning_rate = 0.001
            weight_decay = 1e-3
            patience = 20
        
        # Use BCEWithLogitsLoss for all datasets (better numerical stability)
        # and convert model to logits mode by replacing sigmoid with Identity
        criterion = nn.BCEWithLogitsLoss()
        if hasattr(model, 'sigmoid'):
            model.sigmoid = nn.Identity()
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=8)
        
        best_acc = 0.0
        no_improve = 0
        best_state = None
        
        # 使用tqdm显示训练进度
        pbar = tqdm(range(epochs), desc=f"  CreditNet Training ({dataset_name})", 
                    unit="epoch", ncols=100, leave=True)
        
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device).float()
                optimizer.zero_grad()
                outputs = model(X)
                
                # All datasets now use logits mode
                loss = criterion(outputs.squeeze(), y)
                epoch_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device)
                    outputs = model(X)
                    # All datasets output logits, apply sigmoid for prediction
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.numpy())
            
            val_acc = accuracy_score(val_labels, val_preds)
            scheduler.step(val_acc)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{epoch_loss/len(train_loader):.4f}',
                'val_acc': f'{val_acc:.4f}',
                'best': f'{best_acc:.4f}',
                'patience': f'{no_improve}/{patience}'
            })
            
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                best_state = model.state_dict().copy()
            else:
                no_improve += 1
                if no_improve >= patience:
                    pbar.close()
                    self.logger.info(f"    Early stopping at epoch {epoch+1}")
                    break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
    
    def _train_creditnet_student(self, model: nn.Module, train_loader: DataLoader,
                                  val_loader: DataLoader, dataset_name: str, epochs: int = 100):
        """Train CreditNet student model."""
        self._train_creditnet_teacher(model, train_loader, val_loader, dataset_name, epochs)
    
    def _evaluate_creditnet(self, model: nn.Module, test_loader: DataLoader, 
                            dataset_name: str) -> Dict[str, float]:
        """Evaluate CreditNet model. All models now output logits."""
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                outputs = model(X)
                
                # All datasets output logits, apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)
                
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.squeeze().cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.squeeze().cpu().numpy())
        
        return self._compute_metrics(
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs)
        )
