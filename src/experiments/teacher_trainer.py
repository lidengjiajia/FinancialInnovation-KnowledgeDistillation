#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teacher Model Trainer Module

This module handles:
1. Optuna-based hyperparameter tuning for ensemble models
2. Neural network (CreditNet) training with early stopping
3. Best teacher selection based on validation AUC
4. SHAP value computation and caching
5. Model persistence and reuse
"""

import os
import pickle
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Handle imports
try:
    from src.data.preprocessor import DataPreprocessor
    from src.models.baselines import get_optimized_boosting_models, get_all_baseline_models, train_traditional_baselines
    from src.models.neural import CreditNet
except ImportError:
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from src.data.preprocessor import DataPreprocessor
    from src.models.baselines import get_optimized_boosting_models, get_all_baseline_models, train_traditional_baselines
    from src.models.neural import CreditNet


@dataclass
class TeacherInfo:
    """Information about a trained teacher model."""
    name: str
    model_type: str  # 'ensemble' or 'neural'
    auc_val: float
    auc_test: float
    dataset_name: str
    n_features: int
    hyperparams: Dict[str, Any]
    shap_computed: bool = False


@dataclass
class TeacherCache:
    """Cache for teacher models and their SHAP values."""
    teacher_info: TeacherInfo
    model_path: str
    shap_values_path: Optional[str] = None
    feature_importance_path: Optional[str] = None
    shap_plot_path: Optional[str] = None


class TeacherTrainer:
    """Handles teacher model training, selection, and caching."""
    
    def __init__(
        self,
        data_dir: str = 'data',
        results_dir: str = 'results',
        cache_dir: str = None,
        seed: int = 42,
        n_optuna_trials: int = 50,
        nn_epochs: int = 100,
        use_gpu: bool = True,
        device_id: int = None,
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.cache_dir = cache_dir or os.path.join(results_dir, 'teacher_cache')
        self.seed = seed
        self.n_optuna_trials = n_optuna_trials
        self.nn_epochs = nn_epochs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device_id = device_id
        
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.use_gpu:
            if device_id is not None:
                torch.cuda.set_device(device_id)
                self.device = torch.device(f'cuda:{device_id}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def _get_cache_path(self, dataset_name: str, suffix: str) -> str:
        return os.path.join(self.cache_dir, f'{dataset_name}_{suffix}')
    
    def _get_teacher_cache_path(self, dataset_name: str) -> str:
        return self._get_cache_path(dataset_name, 'teacher_cache.json')
    
    def _get_model_path(self, dataset_name: str, model_name: str) -> str:
        safe_name = model_name.replace('-', '_').replace(' ', '_')
        return self._get_cache_path(dataset_name, f'teacher_{safe_name}.pkl')
    
    def _get_shap_path(self, dataset_name: str) -> str:
        return self._get_cache_path(dataset_name, 'teacher_shap.npz')
    
    def _get_shap_plot_path(self, dataset_name: str) -> str:
        figures_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        return os.path.join(figures_dir, f'shap_{dataset_name}_top10.png')
    
    def _load_data(self, dataset_name: str) -> Optional[Dict]:
        preprocessor = DataPreprocessor(data_dir=self.data_dir, random_state=self.seed)
        loaders = {
            'german': preprocessor.load_german_credit,
            'australian': preprocessor.load_australian_credit,
            'xinwang': preprocessor.load_xinwang_credit,
            'uci': preprocessor.load_uci_credit
        }
        if dataset_name not in loaders:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return None
        return loaders[dataset_name]()
    
    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def check_teacher_cache(self, dataset_name: str, verbose: bool = True) -> Optional[TeacherCache]:
        """
        Check if a valid teacher cache exists.
        
        缓存验证流程:
        1. 检查 {dataset}_teacher_cache.json 是否存在
        2. 读取JSON，验证必要字段
        3. 检查模型文件 (.pkl) 是否存在
        4. 检查SHAP文件 (.npz) 是否存在 (可选但建议)
        5. 检查Excel结果文件是否存在 (可选)
        
        Returns:
            TeacherCache if valid cache exists, None otherwise
        """
        cache_path = self._get_teacher_cache_path(dataset_name)
        
        # Step 1: 检查JSON文件
        if not os.path.exists(cache_path):
            if verbose:
                self.logger.info(f"[{dataset_name}] 无缓存: {cache_path} 不存在")
            return None
        
        try:
            # Step 2: 读取JSON
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Step 3: 检查模型文件
            model_path = cache_data.get('model_path')
            if not model_path or not os.path.exists(model_path):
                self.logger.warning(f"[{dataset_name}] 缓存无效: 模型文件缺失 ({model_path})")
                return None
            
            # Step 4: 检查SHAP文件 (警告但不阻止)
            shap_path = cache_data.get('shap_values_path')
            if shap_path and not os.path.exists(shap_path):
                self.logger.warning(f"[{dataset_name}] SHAP文件缺失: {shap_path}")
                # 不返回None，SHAP可以重新计算
            
            # Step 5: 检查Excel结果 (可选，仅警告)
            excel_path = os.path.join(self.results_dir, f'{dataset_name}_teacher_selection.xlsx')
            if not os.path.exists(excel_path):
                self.logger.info(f"[{dataset_name}] Excel结果缺失，将在蒸馏时重新生成")
            
            teacher_info = TeacherInfo(**cache_data['teacher_info'])
            
            if verbose:
                self.logger.info(f"[{dataset_name}] 缓存有效: {teacher_info.name} (AUC={teacher_info.auc_val:.4f})")
            
            return TeacherCache(
                teacher_info=teacher_info,
                model_path=model_path,
                shap_values_path=cache_data.get('shap_values_path'),
                feature_importance_path=cache_data.get('feature_importance_path'),
                shap_plot_path=cache_data.get('shap_plot_path')
            )
        except Exception as e:
            self.logger.warning(f"[{dataset_name}] 加载缓存失败: {e}")
            return None
    
    def load_teacher_model(self, dataset_name: str) -> Tuple[Any, TeacherInfo, bool]:
        """Load cached teacher model."""
        cache = self.check_teacher_cache(dataset_name)
        if cache is None:
            raise ValueError(f"No valid teacher cache for {dataset_name}")
        
        with open(cache.model_path, 'rb') as f:
            model = pickle.load(f)
        
        is_neural = cache.teacher_info.model_type == 'neural'
        
        if is_neural and hasattr(model, 'to'):
            model = model.to(self.device)
            model.eval()
        
        return model, cache.teacher_info, is_neural
    
    def load_teacher_shap(self, dataset_name: str) -> Optional[Dict]:
        """Load cached SHAP values."""
        cache = self.check_teacher_cache(dataset_name)
        if cache is None or cache.shap_values_path is None:
            return None
        
        if not os.path.exists(cache.shap_values_path):
            return None
        
        try:
            data = np.load(cache.shap_values_path, allow_pickle=True)
            return {
                'shap_values': data['shap_values'],
                'feature_importance': data['feature_importance'],
                'feature_names': data.get('feature_names', None)
            }
        except Exception as e:
            self.logger.warning(f"[{dataset_name}] Failed to load SHAP: {e}")
            return None
    
    def train_ensemble_models(
        self,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        include_traditional: bool = True,
    ) -> Dict[str, Tuple[Any, float, Dict]]:
        """Train all baseline models: Optuna-optimized ensemble + traditional models.
        
        Args:
            dataset_name: Name of the dataset
            X_train, y_train: Training data
            X_val, y_val: Validation data
            include_traditional: Whether to include LR, SVM, DT, KNN, NB models
        """
        self.logger.info(f"  [{dataset_name}] Training baseline models...")
        self.logger.info(f"    - Optuna trials: {self.n_optuna_trials}")
        self.logger.info(f"    - Include traditional models: {include_traditional}")
        
        self._set_seed(self.seed)
        
        # Use the new unified function
        all_results = get_all_baseline_models(
            X_train, y_train, X_val, y_val,
            n_optuna_trials=self.n_optuna_trials,
            use_gpu=self.use_gpu,
            random_state=self.seed,
            include_traditional=include_traditional
        )
        
        results = {}
        for name, opt_result in all_results.items():
            model = opt_result.get('best_model')
            if model is None:
                continue
            
            y_prob = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_prob)
            hyperparams = opt_result.get('best_params', {})
            results[name] = (model, val_auc, hyperparams)
            self.logger.info(f"    {name}: Val AUC = {val_auc:.4f}")
        
        return results
    
    def train_neural_model(
        self,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[CreditNet, float, Dict]:
        """Train CreditNet neural network."""
        self.logger.info(f"  [{dataset_name}] Training CreditNet ({self.nn_epochs} epochs)...")
        
        self._set_seed(self.seed)
        
        input_dim = X_train.shape[1]
        model = CreditNet(input_dim, dataset_type=dataset_name).to(self.device)
        
        # Replace sigmoid with Identity for logits mode
        if hasattr(model, 'sigmoid'):
            model.sigmoid = nn.Identity()
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        batch_size = min(256, len(X_train) // 4)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        criterion = nn.BCEWithLogitsLoss()
        
        best_auc = 0.0
        best_state = None
        patience_counter = 0
        patience = 20
        
        # 使用tqdm显示训练进度
        pbar = tqdm(range(self.nn_epochs), desc=f"  CreditNet Training", 
                    unit="epoch", ncols=100, leave=True)
        
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)  # [B] -> [B, 1]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = model(X_batch)
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    val_probs.extend(probs if probs.ndim > 0 else [probs.item()])
                    val_labels.extend(y_batch.numpy())
            
            val_auc = roc_auc_score(val_labels, val_probs)
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss': f'{epoch_loss/len(train_loader):.4f}',
                'val_auc': f'{val_auc:.4f}',
                'best': f'{best_auc:.4f}',
                'patience': f'{patience_counter}/{patience}'
            })
            scheduler.step(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                self.logger.info(f"    Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        model.eval()
        self.logger.info(f"    CreditNet: Val AUC = {best_auc:.4f}")
        
        hyperparams = {'input_dim': input_dim, 'epochs': self.nn_epochs, 'batch_size': batch_size}
        return model, best_auc, hyperparams
    
    def compute_teacher_shap(
        self,
        model: Any,
        is_neural: bool,
        X_train: np.ndarray,
        feature_names: List[str],
        dataset_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values for the teacher model."""
        self.logger.info(f"  [{dataset_name}] Computing SHAP values...")
        
        try:
            import shap
            
            # 检测模型类型
            model_class_name = type(model).__name__
            tree_based_models = ['RandomForestClassifier', 'GradientBoostingClassifier', 
                                 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
                                 'DecisionTreeClassifier', 'ExtraTreesClassifier']
            is_tree_based = any(name in model_class_name for name in tree_based_models)
            
            if is_neural or not is_tree_based:
                # 使用 KernelExplainer 处理神经网络和非树模型 (LR, SVM, KNN, NB)
                self.logger.info(f"    Using KernelExplainer for {model_class_name}")
                
                def model_predict(x):
                    if is_neural:
                        model.eval()
                        with torch.no_grad():
                            x_tensor = torch.FloatTensor(x).to(self.device)
                            outputs = model(x_tensor)
                            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                        return probs
                    else:
                        return model.predict_proba(x)[:, 1]
                
                np.random.seed(self.seed)
                n_bg = min(100, len(X_train))
                bg_idx = np.random.choice(len(X_train), n_bg, replace=False)
                
                explainer = shap.KernelExplainer(model_predict, X_train[bg_idx])
                n_eval = min(500, len(X_train))
                shap_values = explainer.shap_values(X_train[:n_eval])
                
                if len(shap_values) < len(X_train):
                    avg_shap = shap_values.mean(axis=0)
                    padding = np.tile(avg_shap, (len(X_train) - len(shap_values), 1))
                    shap_values = np.vstack([shap_values, padding])
            else:
                # 使用 TreeExplainer 处理树模型 (RF, GBDT, XGBoost, LightGBM, CatBoost)
                self.logger.info(f"    Using TreeExplainer for {model_class_name}")
                explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(X_train, check_additivity=False)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                shap_values = np.asarray(shap_values)
                
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            
            feature_importance = np.abs(shap_values).mean(axis=0)
            self.logger.info(f"    SHAP values shape: {shap_values.shape}")
            
            return shap_values, feature_importance
            
        except Exception as e:
            self.logger.error(f"  [{dataset_name}] SHAP computation failed: {e}")
            n_features = X_train.shape[1]
            return np.ones((len(X_train), n_features)), np.ones(n_features) / n_features
    
    def generate_shap_plot(
        self,
        feature_importance: np.ndarray,
        feature_names: List[str],
        dataset_name: str,
        teacher_name: str,
        top_k: int = 10
    ) -> str:
        """Generate and save SHAP feature importance plot."""
        self.logger.info(f"  [{dataset_name}] Generating SHAP importance plot...")
        
        feature_importance = np.asarray(feature_importance).flatten()
        
        if len(feature_importance) != len(feature_names):
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        sorted_idx = np.argsort(-feature_importance)
        top_idx = sorted_idx[:top_k]
        top_features = [str(feature_names[i]) for i in top_idx]
        top_vals = [float(feature_importance[i]) for i in top_idx]
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.8, len(top_features)))
        
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_vals, color=colors, edgecolor='#2C3E50', linewidth=1.2, alpha=0.85)
        
        for i, (bar, val) in enumerate(zip(bars, top_vals)):
            ax.text(val + max(top_vals) * 0.01, i, f'{val:.4f}', 
                   va='center', fontsize=10, fontweight='bold', color='#2C3E50')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11, fontweight='600')
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|', fontsize=13, fontweight='bold', color='#2C3E50')
        ax.set_title(f'SHAP Feature Importance - {dataset_name.upper()} (Teacher: {teacher_name})',
                    fontsize=16, fontweight='bold', color='#1A252F', pad=20)
        
        ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8, color='#7F8C8D')
        ax.set_axisbelow(True)
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('white')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#2C3E50')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        save_path = self._get_shap_plot_path(dataset_name)
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
        plt.close(fig)
        
        self.logger.info(f"    SHAP plot saved to: {save_path}")
        return save_path
    
    def train_and_select_teacher(
        self,
        dataset_name: str,
        force_retrain: bool = False,
        compute_shap: bool = True,
    ) -> TeacherCache:
        """Train all candidate teachers, select the best, and cache results."""
        if not force_retrain:
            cache = self.check_teacher_cache(dataset_name)
            if cache is not None:
                self.logger.info(f"[{dataset_name}] Using cached teacher: {cache.teacher_info.name}")
                return cache
        
        self.logger.info(f"[{dataset_name}] Training teacher models...")
        
        data = self._load_data(dataset_name)
        if data is None:
            raise ValueError(f"Failed to load data for {dataset_name}")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
        
        # Train ensemble models
        ensemble_results = self.train_ensemble_models(dataset_name, X_train, y_train, X_val, y_val)
        
        # Train neural model
        neural_model, neural_auc, neural_params = self.train_neural_model(dataset_name, X_train, y_train, X_val, y_val)
        
        # Select best teacher
        best_name = None
        best_model = None
        best_auc = 0.0
        best_params = {}
        is_neural = False
        
        for name, (model, auc, params) in ensemble_results.items():
            if auc > best_auc:
                best_auc = auc
                best_name = name
                best_model = model
                best_params = params
                is_neural = False
        
        if neural_auc > best_auc:
            best_auc = neural_auc
            best_name = 'CreditNet'
            best_model = neural_model
            best_params = neural_params
            is_neural = True
        
        self.logger.info(f"[{dataset_name}] Best teacher: {best_name} (Val AUC = {best_auc:.4f})")
        
        # ====================================================================
        # 保存所有训练过的模型 (方便下次直接使用)
        # ====================================================================
        self.logger.info(f"[{dataset_name}] Saving all trained models...")
        all_models_info = {}
        
        # 保存所有ensemble模型
        for name, (model, auc, params) in ensemble_results.items():
            model_path = self._get_model_path(dataset_name, name)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            all_models_info[name] = {'path': model_path, 'auc': auc, 'type': 'ensemble'}
            self.logger.info(f"    Saved {name}: {model_path}")
        
        # 保存neural模型
        neural_path = self._get_model_path(dataset_name, 'CreditNet')
        with open(neural_path, 'wb') as f:
            pickle.dump(neural_model.cpu(), f)
        neural_model.to(self.device)
        all_models_info['CreditNet'] = {'path': neural_path, 'auc': neural_auc, 'type': 'neural'}
        self.logger.info(f"    Saved CreditNet: {neural_path}")
        
        # 保存所有模型的索引
        all_models_index_path = self._get_cache_path(dataset_name, 'all_models_index.json')
        with open(all_models_index_path, 'w') as f:
            json.dump(all_models_info, f, indent=2)
        
        # ====================================================================
        # Evaluate on test set
        # ====================================================================
        if is_neural:
            best_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                outputs = best_model(X_test_tensor)
                test_probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        else:
            test_probs = best_model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, test_probs)
        self.logger.info(f"[{dataset_name}] Test AUC: {test_auc:.4f}")
        
        # Create TeacherInfo
        teacher_info = TeacherInfo(
            name=best_name,
            model_type='neural' if is_neural else 'ensemble',
            auc_val=best_auc,
            auc_test=test_auc,
            dataset_name=dataset_name,
            n_features=X_train.shape[1],
            hyperparams=best_params,
            shap_computed=False
        )
        
        # 使用已保存的最佳模型路径
        model_path = self._get_model_path(dataset_name, best_name)
        
        # Compute SHAP
        shap_path = None
        shap_plot_path = None
        if compute_shap:
            shap_values, feature_importance = self.compute_teacher_shap(
                best_model, is_neural, X_train, feature_names, dataset_name
            )
            
            shap_path = self._get_shap_path(dataset_name)
            np.savez(shap_path, shap_values=shap_values, feature_importance=feature_importance, feature_names=feature_names)
            
            shap_plot_path = self.generate_shap_plot(feature_importance, feature_names, dataset_name, best_name)
            teacher_info.shap_computed = True
        
        # Save cache metadata
        cache = TeacherCache(
            teacher_info=teacher_info,
            model_path=model_path,
            shap_values_path=shap_path,
            shap_plot_path=shap_plot_path
        )
        
        cache_path = self._get_teacher_cache_path(dataset_name)
        with open(cache_path, 'w') as f:
            json.dump({
                'teacher_info': asdict(teacher_info),
                'model_path': model_path,
                'shap_values_path': shap_path,
                'shap_plot_path': shap_plot_path
            }, f, indent=2)
        
        # Save teacher selection summary
        self._save_teacher_summary(dataset_name, teacher_info, ensemble_results, neural_auc)
        
        return cache
    
    def _get_model_category(self, model_name: str) -> str:
        """Determine model category based on name."""
        if model_name in ['LR-Ridge', 'LR-Lasso', 'LR-ElasticNet']:
            return 'linear'
        elif model_name in ['SVM-RBF', 'SVM-Linear']:
            return 'kernel'
        elif model_name == 'DT':
            return 'tree'
        elif model_name in ['KNN']:
            return 'instance-based'
        elif model_name in ['NB']:
            return 'probabilistic'
        elif model_name in ['RF-Tuned', 'GBDT-Tuned', 'XGBoost-Tuned', 'LightGBM-Tuned', 'CatBoost-Tuned']:
            return 'ensemble-optimized'
        elif model_name == 'CreditNet':
            return 'neural'
        else:
            return 'ensemble'
    
    def _save_teacher_summary(self, dataset_name: str, teacher_info: TeacherInfo, ensemble_results: Dict, neural_auc: float):
        """Save teacher selection summary to Excel."""
        rows = []
        
        for name, (model, auc, params) in ensemble_results.items():
            category = self._get_model_category(name)
            rows.append({
                'model': name,
                'type': category,
                'val_auc': auc,
                'is_best': name == teacher_info.name,
                'hyperparams': str(params)
            })
        
        rows.append({
            'model': 'CreditNet',
            'type': 'neural',
            'val_auc': neural_auc,
            'is_best': teacher_info.name == 'CreditNet',
            'hyperparams': str(teacher_info.hyperparams) if teacher_info.name == 'CreditNet' else ''
        })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('val_auc', ascending=False)
        
        path = os.path.join(self.results_dir, f'{dataset_name}_teacher_selection.xlsx')
        df.to_excel(path, index=False)
        self.logger.info(f"[{dataset_name}] Teacher selection saved to {path}")


def train_teachers_parallel(
    datasets: List[str],
    data_dir: str = 'data',
    results_dir: str = 'results',
    seed: int = 42,
    n_optuna_trials: int = 50,
    nn_epochs: int = 100,
    use_gpu: bool = True,
    force_retrain: bool = False,
    max_workers: int = None,
) -> Dict[str, TeacherCache]:
    """Train teachers for multiple datasets in parallel."""
    logger = logging.getLogger(__name__)
    
    if max_workers is None:
        max_workers = 1 if use_gpu else min(len(datasets), 4)
    
    results = {}
    
    if max_workers == 1:
        trainer = TeacherTrainer(
            data_dir=data_dir,
            results_dir=results_dir,
            seed=seed,
            n_optuna_trials=n_optuna_trials,
            nn_epochs=nn_epochs,
            use_gpu=use_gpu
        )
        
        for dataset in datasets:
            try:
                cache = trainer.train_and_select_teacher(dataset, force_retrain=force_retrain)
                results[dataset] = cache
            except Exception as e:
                logger.error(f"Failed to train teacher for {dataset}: {e}")
    else:
        def train_one(dataset_name):
            trainer = TeacherTrainer(
                data_dir=data_dir,
                results_dir=results_dir,
                seed=seed,
                n_optuna_trials=n_optuna_trials,
                nn_epochs=nn_epochs,
                use_gpu=False
            )
            return dataset_name, trainer.train_and_select_teacher(dataset_name, force_retrain=force_retrain)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(train_one, ds): ds for ds in datasets}
            
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    ds_name, cache = future.result()
                    results[ds_name] = cache
                except Exception as e:
                    logger.error(f"Failed to train teacher for {dataset}: {e}")
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    
    datasets = ['german']
    results = train_teachers_parallel(datasets, n_optuna_trials=10, nn_epochs=20, force_retrain=True)
    
    for ds, cache in results.items():
        print(f"{ds}: {cache.teacher_info.name} (AUC={cache.teacher_info.auc_val:.4f})")
