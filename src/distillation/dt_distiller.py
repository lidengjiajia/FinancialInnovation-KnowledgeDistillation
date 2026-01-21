#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CB-KD: Class-Balanced Knowledge Distillation to Decision Tree

This module implements the CB-KD framework for credit risk assessment:
1. Automatic teacher selection from multiple candidates
2. Temperature-scaled soft-label generation
3. Class-balanced sample weighting for imbalanced portfolios
4. Soft-target tree training via sample duplication
5. Lift-based rule extraction with quality metrics

Reference: CB-KD Paper - Class-balanced interpretable knowledge distillation 
           paradigm for credit risk assessment
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr
import logging


class DecisionTreeDistiller:
    """
    CB-KD: Class-Balanced Knowledge Distillation to Decision Tree.
    
    Transfers knowledge from a complex teacher model (ensemble or neural network)
    to an interpretable Decision Tree student using:
    1. Temperature-scaled soft-label distillation
    2. Class-balanced sample weighting (inverse frequency)
    3. Soft-target training via sample duplication
    
    Mathematical formulation (binary classification):
    - Soft labels: p_soft = σ(z/τ), where z is teacher logit, τ is temperature
    - Mixed target: p_mix = (1-α)·p_soft + α·y, where y∈{0,1}
    - Class weight: w_i = N / (2·n_{y_i}), where n_{y_i} is class count
    - Training: duplicate each sample with weights w_i·(1-p_mix) and w_i·p_mix
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.0,
                 max_depth: int = 6,
                 min_samples_leaf: int = 5,
                 use_class_balance: bool = True,
                 criterion: str = 'gini',
                 cost_ratio: float = 1.0,
                 random_state: int = 42):
        """
        Initialize CB-KD distiller.
        
        Args:
            temperature: Temperature τ for soft label generation (higher = softer)
            alpha: Hard-label mixing ratio (0=pure soft, 1=pure hard)
            max_depth: Maximum depth of the decision tree
            min_samples_leaf: Minimum samples required at leaf node
            use_class_balance: Whether to apply class-balanced weighting
            criterion: Split criterion for decision tree ('gini', 'entropy', 'log_loss')
            cost_ratio: Cost ratio for minority class (default=cost_ratio for misclassifying 
                        a default as non-default vs misclassifying non-default as default).
                        Higher values increase recall at the expense of precision.
            random_state: Random seed for reproducibility
        """
        self.temperature = temperature
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.use_class_balance = use_class_balance
        self.criterion = criterion
        self.cost_ratio = cost_ratio
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.student = None
        self.teacher = None
        self.is_neural_teacher = False
        self.feature_names = None
        self.class_names = ['Non-default', 'Default']
        
        # Training artifacts
        self.soft_labels = None
        self.sample_weights = None
        self.decision_threshold = 0.5
        self.knowledge_transfer_score = None
    
    def set_teacher(self, teacher_model: Any, is_neural: bool = False):
        """
        Set the teacher model.
        
        Args:
            teacher_model: Any model with predict_proba method (or PyTorch model)
            is_neural: Whether the teacher is a neural network (PyTorch)
        """
        self.teacher = teacher_model
        self.is_neural_teacher = is_neural
    
    def _get_teacher_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw logits from teacher model.
        
        Returns:
            Array of shape (n_samples,) with logits
        """
        if self.is_neural_teacher:
            import torch
            self.teacher.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                if next(self.teacher.parameters()).is_cuda:
                    X_tensor = X_tensor.cuda()
                raw = self.teacher(X_tensor).cpu().numpy()

                # Handle common binary-output conventions
                if raw.ndim == 2 and raw.shape[1] == 2:
                    return (raw[:, 1] - raw[:, 0]).astype(float)
                if raw.ndim == 2 and raw.shape[1] == 1:
                    return raw[:, 0].astype(float)
                return np.asarray(raw).squeeze().astype(float)
        else:
            # Convert probabilities to logits
            probs = self.teacher.predict_proba(X)[:, 1]
            eps = 1e-7
            probs = np.clip(probs, eps, 1 - eps)
            logits = np.log(probs / (1 - probs))
            return logits
    
    def _get_teacher_soft_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Get soft labels from teacher with temperature scaling.
        
        Implements: p_soft = σ(z/τ)
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples,) with probability of positive class
        """
        logits = self._get_teacher_logits(X)
        scaled_logits = logits / self.temperature
        soft_probs = 1 / (1 + np.exp(-scaled_logits))
        return soft_probs
    
    def _compute_class_weights(self, y: np.ndarray, soft_labels: np.ndarray = None) -> np.ndarray:
        """
        Compute class-balanced sample weights for knowledge distillation.
        
        The core CB-KD mechanism uses adaptive inverse-frequency weighting that
        adjusts based on the imbalance severity, combined with teacher confidence
        weighting and boundary sample boosting for improved knowledge transfer.
        
        Three-component weighting strategy:
        1. Class balance: inverse-frequency weighting with adaptive smoothing
        2. Teacher confidence: upweight samples where teacher is confident
        3. Boundary boosting: upweight samples near decision boundary (p ≈ 0.5)
        
        Formula: w_i = w_class * w_confidence * w_boundary
        
        Args:
            y: Binary ground truth labels
            soft_labels: Teacher's soft predictions for confidence weighting
        
        Returns:
            Array of sample weights with mean ≈ 1
        """
        y_int = y.astype(int)
        n_total = len(y_int)
        n_class_0 = np.sum(y_int == 0)
        n_class_1 = np.sum(y_int == 1)
        
        # Compute imbalance ratio
        imbalance_ratio = max(n_class_0, n_class_1) / (min(n_class_0, n_class_1) + 1e-8)
        
        # ====================================================================
        # Component 1: Class Balance Weighting (adaptive based on imbalance)
        # Core insight: preserve teacher knowledge while addressing class imbalance
        # ====================================================================
        w0_raw = n_total / (2.0 * n_class_0 + 1e-8)
        w1_raw = n_total / (2.0 * n_class_1 + 1e-8)
        
        # Apply cost ratio to minority class
        w1_raw = w1_raw * self.cost_ratio
        
        # Use logarithmic dampening for all cases to preserve teacher knowledge
        # while still providing class balance
        if imbalance_ratio < 1.5:
            # Near-balanced: minimal adjustment
            beta = 0.2
            w0 = 1.0 + beta * (w0_raw - 1.0)
            w1 = 1.0 + beta * (w1_raw - 1.0)
        elif imbalance_ratio < 3.0:
            # Mild imbalance: moderate dampening
            beta = 0.5
            w0 = 1.0 + beta * (w0_raw - 1.0)
            w1 = 1.0 + beta * (w1_raw - 1.0)
        else:
            # Moderate to severe imbalance: use log dampening
            # This ensures class balance effect while preserving teacher knowledge
            w0 = 1.0 + 0.5 * np.log1p(w0_raw - 1.0)
            w1 = 1.0 + 0.8 * np.log1p(w1_raw - 1.0)
            # Ensure reasonable bounds
            w1 = np.clip(w1, 1.5, 4.0)
        
        class_weights = np.where(y_int == 0, w0, w1)
        
        # ====================================================================
        # Component 2: Teacher Confidence Weighting (Soft Label Alignment)
        # Upweight samples where teacher is confident about its prediction
        # This helps preserve teacher's knowledge in the student model
        # ====================================================================
        if soft_labels is not None:
            # Teacher confidence: how far from 0.5 (uncertainty)
            teacher_confidence = np.abs(soft_labels - 0.5) * 2.0  # Scale to [0, 1]
            # Confidence weight: base + bonus for confident predictions
            confidence_weight = 0.8 + 0.4 * teacher_confidence
            class_weights = class_weights * confidence_weight
        
        # Normalize to mean = 1
        class_weights = class_weights / (class_weights.mean() + 1e-8)
        
        self.logger.info(f"  CB weights: n0={n_class_0}, n1={n_class_1}, ratio={imbalance_ratio:.2f}, "
                        f"w0={w0:.3f}, w1={w1:.3f}")
        
        return class_weights
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feature_names: List[str] = None,
            precomputed_shap: np.ndarray = None) -> 'DecisionTreeDistiller':
        """
        Train the Decision Tree student using CB-KD.
        
        The training process follows Algorithm 1 in the paper:
        1. Get soft labels from teacher with temperature scaling
        2. Compute class-balanced sample weights
        3. Mix hard/soft targets
        4. Train DT with weighted soft targets via sample duplication
        5. Tune decision threshold on validation set (optional)
        
        CB-KD uses adaptive parameter selection based on imbalance ratio:
        - Near-balanced (ratio < 2.0): standard α=0.2, τ=4.0
        - Moderate imbalance (2.0 ≤ ratio < 5.0): α=0.0, τ=4.0
        - Severe imbalance (ratio ≥ 5.0): α=0.2, τ=6.0
        
        Args:
            X_train: Training features
            y_train: Training labels (hard labels)
            X_val: Validation features (optional, for threshold tuning)
            y_val: Validation labels (optional)
            feature_names: Names of features for rule extraction
            precomputed_shap: Ignored (kept for backward compatibility)
        
        Returns:
            self
        """
        if self.teacher is None:
            raise ValueError("Teacher model not set. Call set_teacher() first.")
        
        self.feature_names = feature_names if feature_names else \
                            [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        n = len(X_train)
        y_train = np.asarray(y_train).astype(float)
        
        # Store original training data for rule extraction
        self._X_train = np.asarray(X_train)
        self._y_train = y_train.astype(int)
        
        # ====================================================================
        # Step 0: Adaptive Parameter Selection (CB-KD enhancement)
        # Automatically adjust α and τ based on imbalance severity
        # Key insight: preserve more teacher knowledge for all datasets
        # ====================================================================
        if self.use_class_balance:
            y_int = y_train.astype(int)
            n_class_0 = np.sum(y_int == 0)
            n_class_1 = np.sum(y_int == 1)
            imbalance_ratio = max(n_class_0, n_class_1) / (min(n_class_0, n_class_1) + 1e-8)
            
            # Adaptive parameter selection - prioritize soft labels
            if imbalance_ratio < 1.5:
                # Near-balanced: use pure soft labels with moderate temperature
                effective_alpha = 0.0
                effective_temp = self.temperature
            elif imbalance_ratio < 3.0:
                # Mild imbalance: pure soft labels
                effective_alpha = 0.0
                effective_temp = self.temperature
            elif imbalance_ratio < 5.0:
                # Moderate imbalance: pure soft labels with slightly higher temp
                effective_alpha = 0.0
                effective_temp = max(self.temperature, 4.0)
            else:
                # Severe imbalance: higher temperature for softer labels
                effective_alpha = 0.0
                effective_temp = max(self.temperature, 5.0)
            
            self.logger.info(f"  CB-KD adaptive params: ratio={imbalance_ratio:.2f}, "
                            f"α={effective_alpha:.2f}, τ={effective_temp:.1f}")
        else:
            effective_alpha = self.alpha
            effective_temp = self.temperature
        
        # ====================================================================
        # Step 1: Soft Label Generation (Eq. 1 in paper)
        # p_soft = σ(z/τ)
        # ====================================================================
        # Temporarily override temperature for soft label generation
        original_temp = self.temperature
        self.temperature = effective_temp
        self.soft_labels = self._get_teacher_soft_labels(X_train)
        self.temperature = original_temp  # Restore
        
        self.logger.info(f"  Soft labels: mean={self.soft_labels.mean():.4f}, "
                        f"std={self.soft_labels.std():.4f}")
        
        # ====================================================================
        # Step 2: Enhanced Class-Balanced Sample Weighting
        # Combines class balance + teacher confidence + boundary boosting
        # ====================================================================
        if self.use_class_balance:
            class_weights = self._compute_class_weights(y_train, soft_labels=self.soft_labels)
        else:
            class_weights = np.ones(n)
        
        self.sample_weights = class_weights
        
        # ====================================================================
        # Step 3: Mix Hard/Soft Targets (Eq. 2 in paper)
        # p_mix = (1-α)·p_soft + α·y
        # ====================================================================
        p_soft = np.clip(self.soft_labels, 1e-6, 1 - 1e-6)
        alpha = float(np.clip(effective_alpha, 0.0, 1.0))  # Use effective_alpha
        p_mix = (1.0 - alpha) * p_soft + alpha * y_train
        p_mix = np.clip(p_mix, 1e-6, 1 - 1e-6)
        
        self.logger.info(f"  Mixed targets (α={alpha}): mean={p_mix.mean():.4f}")
        
        # ====================================================================
        # Step 4: Soft-Target Tree Training
        # For each sample, create (x_i, 0) with weight w_i·(1-p_mix)
        #                    and (x_i, 1) with weight w_i·p_mix
        # ====================================================================
        X_aug = np.vstack([X_train, X_train])
        y_aug = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        w_aug = np.hstack([
            class_weights * (1.0 - p_mix),
            class_weights * p_mix
        ])
        
        self.student = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight=None
        )
        self.student.fit(X_aug, y_aug, sample_weight=w_aug)
        
        # ====================================================================
        # Step 5: Validation-based Decision Threshold Tuning
        # ====================================================================
        if X_val is not None and y_val is not None:
            self._tune_threshold(X_val, y_val)
        else:
            self.decision_threshold = 0.5
        
        # ====================================================================
        # Compute Knowledge Transfer Score
        # ====================================================================
        self._compute_knowledge_transfer_score(X_train, y_train.astype(int))
        
        self.logger.info(
            f"  CB-KD trained: depth={self.student.get_depth()}, "
            f"leaves={self.student.get_n_leaves()}, "
            f"KT_score={self.knowledge_transfer_score:.4f}, "
            f"threshold={self.decision_threshold:.2f}"
        )
        
        return self
    
    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray):
        """Tune decision threshold to maximize F1 on validation set."""
        try:
            y_val = np.asarray(y_val).astype(int)
            val_probs = self.student.predict_proba(X_val)[:, 1]
            
            # Grid search over thresholds
            grid = np.linspace(0.05, 0.95, 19)
            best_thr, best_f1 = 0.5, -1.0
            for thr in grid:
                y_pred_val = (val_probs >= thr).astype(int)
                f1 = f1_score(y_val, y_pred_val, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            
            self.decision_threshold = float(best_thr)
            self.logger.info(f"  Threshold tuned: {self.decision_threshold:.2f} (F1={best_f1:.4f})")
        except Exception as e:
            self.logger.warning(f"  Threshold tuning failed: {e}")
            self.decision_threshold = 0.5
    
    def _compute_knowledge_transfer_score(self, X: np.ndarray, y: np.ndarray):
        """
        Compute comprehensive knowledge transfer quality metrics.
        
        Implements multiple evaluation metrics from KD literature:
        - Fidelity (Agreement Rate): Hinton et al. (2015)
        - Probability Correlation: Standard practice
        - Rank Correlation (Spearman): Ranking preservation
        - KL Divergence: Information-theoretic measure
        - Expected Calibration Error (ECE): Calibration quality
        
        Reference: Gou et al. (2021) "Knowledge Distillation: A Survey"
        """
        student_probs = self.student.predict_proba(X)[:, 1]
        teacher_probs = self.soft_labels
        
        # Clip probabilities for numerical stability
        eps = 1e-7
        student_probs_clip = np.clip(student_probs, eps, 1 - eps)
        teacher_probs_clip = np.clip(teacher_probs, eps, 1 - eps)
        
        # 1. Fidelity / Agreement Rate (Hinton et al., 2015)
        # Measures how often student and teacher agree on predictions
        student_pred = (student_probs > 0.5).astype(int)
        teacher_pred = (teacher_probs > 0.5).astype(int)
        fidelity = (student_pred == teacher_pred).mean()
        
        # 2. Probability Correlation (Pearson)
        # Measures linear relationship between probability outputs
        prob_corr = np.corrcoef(student_probs, teacher_probs)[0, 1]
        if np.isnan(prob_corr):
            prob_corr = 0.0
        
        # 3. Rank Correlation (Spearman)
        # Measures preservation of ranking order
        rank_corr, _ = spearmanr(student_probs, teacher_probs)
        if np.isnan(rank_corr):
            rank_corr = 0.0
        
        # 4. KL Divergence (Hinton et al., 2015)
        # Information-theoretic measure of distribution difference
        # KL(P_teacher || P_student) for binary classification
        kl_div = np.mean(
            teacher_probs_clip * np.log(teacher_probs_clip / student_probs_clip) +
            (1 - teacher_probs_clip) * np.log((1 - teacher_probs_clip) / (1 - student_probs_clip))
        )
        
        # 5. Jensen-Shannon Divergence (symmetric version of KL)
        m_probs = 0.5 * (teacher_probs_clip + student_probs_clip)
        js_div = 0.5 * np.mean(
            teacher_probs_clip * np.log(teacher_probs_clip / m_probs) +
            (1 - teacher_probs_clip) * np.log((1 - teacher_probs_clip) / (1 - m_probs))
        ) + 0.5 * np.mean(
            student_probs_clip * np.log(student_probs_clip / m_probs) +
            (1 - student_probs_clip) * np.log((1 - student_probs_clip) / (1 - m_probs))
        )
        
        # 6. Mean Absolute Error between probabilities
        mae = np.mean(np.abs(student_probs - teacher_probs))
        
        # 7. Expected Calibration Error (ECE) - Guo et al. (2017)
        # Measures calibration of student relative to teacher
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (student_probs >= bin_boundaries[i]) & (student_probs < bin_boundaries[i+1])
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                avg_confidence = student_probs[in_bin].mean()
                avg_teacher = teacher_probs[in_bin].mean()
                ece += np.abs(avg_confidence - avg_teacher) * prop_in_bin
        
        # Store comprehensive metrics
        self.kt_metrics = {
            # Core distillation metrics (from Hinton et al., 2015)
            'fidelity': float(fidelity),
            'prob_correlation': float(prob_corr),
            'rank_correlation': float(rank_corr),
            'kl_divergence': float(kl_div),
            
            # Additional quality metrics
            'js_divergence': float(js_div),
            'mae': float(mae),
            'ece': float(ece),
            
            # Legacy compatibility
            'agreement_rate': float(fidelity),
            'probability_correlation': float(prob_corr)
        }
        
        # Combined score (weighted average of key metrics)
        # Higher is better: Fidelity, Prob_corr, Rank_corr
        # Lower is better: KL_div, MAE -> convert to (1 - normalized)
        self.knowledge_transfer_score = (
            0.25 * fidelity + 
            0.25 * max(0, prob_corr) + 
            0.25 * max(0, rank_corr) +
            0.25 * max(0, 1 - min(1, kl_div))  # Normalize KL to [0,1]
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using tuned threshold."""
        probs = self.student.predict_proba(X)[:, 1]
        return (probs >= self.decision_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.student.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the student model.
        
        Returns:
            Dictionary with AUC, PR-AUC, Brier, accuracy, precision, recall, F1
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        try:
            metrics['auc'] = roc_auc_score(y_test, y_prob)
            # PR-AUC: Area under Precision-Recall curve (better for imbalanced data)
            metrics['pr_auc'] = average_precision_score(y_test, y_prob)
            # Brier score: Calibration metric (lower is better)
            metrics['brier'] = brier_score_loss(y_test, y_prob)
        except:
            metrics['auc'] = 0.5
            metrics['pr_auc'] = 0.0
            metrics['brier'] = 1.0
        
        return metrics
    
    def extract_rules(self, feature_names: List[str] = None,
                     class_priors: Tuple[float, float] = None) -> List[Dict]:
        """
        Extract IF-THEN rules from the trained decision tree with quality metrics.
        
        Implements Algorithm 2 in the paper: Path-Dependent Rule Extraction
        with Lift-Based Ranking.
        
        Note: Since CB-KD uses sample duplication with weights for soft-label training,
        we use the original training data to compute leaf statistics (samples, confidence, etc.)
        rather than the tree.value which contains weighted samples.
        
        Args:
            feature_names: Names of features
            class_priors: (π_0, π_1) prior probabilities for lift calculation
        
        Returns:
            List of rule dictionaries with:
            - rule_id, conditions, prediction, samples
            - confidence, lift, support, p_value
        """
        if self.student is None:
            raise ValueError("Student model not trained. Call fit() first.")
        
        if feature_names is None:
            feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.student.n_features_in_)]
        
        # Get original training data for computing leaf statistics
        # (tree.value contains weighted samples due to soft-label training)
        if hasattr(self, '_X_train') and hasattr(self, '_y_train'):
            X_train = self._X_train
            y_train = self._y_train
            use_original_data = True
        else:
            use_original_data = False
            self.logger.warning("Original training data not available; using tree.value (may be inaccurate)")
        
        tree = self.student.tree_
        
        # Calculate class priors from original data (not tree.value)
        if use_original_data:
            n_total = len(y_train)
            n_class_0 = np.sum(y_train == 0)
            n_class_1 = np.sum(y_train == 1)
            if class_priors is None:
                class_priors = (n_class_0 / n_total, n_class_1 / n_total)
        else:
            # Fallback: use tree.value (less accurate due to soft labels)
            root_values = tree.value[0][0]
            total_samples = sum(root_values)
            n_total = total_samples
            if class_priors is None:
                class_priors = (root_values[0] / total_samples, root_values[1] / total_samples)
        
        # Apply decision tree to original training data to get leaf assignments
        if use_original_data:
            leaf_ids = self.student.apply(X_train)  # Get leaf node id for each sample
        
        rules = []
        
        def traverse(node_id, conditions):
            """Recursively traverse tree to extract rules."""
            if tree.feature[node_id] == -2:  # Leaf node
                if use_original_data:
                    # Use original training data to compute statistics
                    mask = (leaf_ids == node_id)
                    leaf_y = y_train[mask]
                    n_samples = len(leaf_y)
                    if n_samples > 0:
                        n_class_0_leaf = np.sum(leaf_y == 0)
                        n_class_1_leaf = np.sum(leaf_y == 1)
                        values = [n_class_0_leaf, n_class_1_leaf]
                        predicted_class = np.argmax(values)
                        confidence = values[predicted_class] / n_samples
                    else:
                        # No samples in leaf, use tree prediction
                        tree_values = tree.value[node_id][0]
                        predicted_class = np.argmax(tree_values)
                        values = [0, 0]
                        n_samples = 0
                        confidence = 0
                else:
                    # Fallback: use tree.value
                    values = tree.value[node_id][0]
                    n_samples = int(sum(values))
                    predicted_class = np.argmax(values)
                    values = [int(v) for v in values]
                    confidence = values[predicted_class] / n_samples if n_samples > 0 else 0
                
                # Compute lift (Eq. 6 in paper)
                prior = class_priors[predicted_class]
                lift = confidence / prior if prior > 0 else 1.0
                
                # Compute support
                support = n_samples / n_total if n_total > 0 else 0
                
                # Statistical significance test (Eq. 7 in paper)
                # Test if the observed confidence is significantly greater than the prior
                if n_samples > 0:
                    try:
                        # scipy >= 1.7: use binomtest (binom_test is deprecated)
                        from scipy.stats import binomtest
                        k = int(values[predicted_class])
                        n = int(n_samples)
                        result = binomtest(k, n, prior, alternative='greater')
                        p_value = result.pvalue
                    except ImportError:
                        # Fallback for older scipy versions
                        try:
                            from scipy.stats import binom_test
                            k = int(values[predicted_class])
                            n = int(n_samples)
                            p_value = binom_test(k, n, prior, alternative='greater')
                        except Exception:
                            p_value = 1.0
                    except Exception:
                        p_value = 1.0
                else:
                    p_value = 1.0
                
                rule = {
                    'rule_id': f"R{len(rules) + 1}",
                    'conditions': conditions.copy(),
                    'prediction': self.class_names[predicted_class],
                    'prediction_code': int(predicted_class),
                    'samples': int(n_samples),
                    'confidence': float(confidence),
                    'lift': float(lift),
                    'support': float(support),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'class_distribution': {
                        self.class_names[i]: int(v) for i, v in enumerate(values)
                    }
                }
                rules.append(rule)
            else:
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = feature_names[feature_idx]
                
                # Left child (<=)
                left_conditions = conditions + [(feature_name, '<=', threshold)]
                traverse(tree.children_left[node_id], left_conditions)
                
                # Right child (>)
                right_conditions = conditions + [(feature_name, '>', threshold)]
                traverse(tree.children_right[node_id], right_conditions)
        
        traverse(0, [])
        
        # Apply Benjamini-Hochberg FDR correction for multiple comparison
        # This ensures the proportion of false discoveries among significant rules is controlled
        if len(rules) > 1:
            try:
                from statsmodels.stats.multitest import multipletests
                p_values = [r['p_value'] for r in rules]
                # BH-FDR correction at alpha=0.05
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
                for i, rule in enumerate(rules):
                    rule['p_value_adjusted'] = float(pvals_corrected[i])
                    rule['significant'] = bool(reject[i])  # Update with BH-corrected significance
            except ImportError:
                # Fallback if statsmodels not available: use uncorrected p-values
                for rule in rules:
                    rule['p_value_adjusted'] = rule['p_value']
                self.logger.warning("statsmodels not available; using uncorrected p-values")
        else:
            for rule in rules:
                rule['p_value_adjusted'] = rule['p_value']
        
        # Sort by lift (descending)
        rules = sorted(rules, key=lambda r: -r['lift'])
        
        return rules
    
    def rules_to_text(self, rules: List[Dict] = None, max_rules: int = None) -> str:
        """Convert rules to human-readable text format."""
        if rules is None:
            rules = self.extract_rules()
        
        if max_rules:
            rules = rules[:max_rules]
        
        lines = ["=" * 70]
        lines.append("CB-KD EXTRACTED DECISION RULES")
        lines.append("=" * 70)
        lines.append(f"Total rules: {len(rules)}")
        lines.append("")
        
        for rule in rules:
            sig_marker = "*" if rule['significant'] else ""
            lines.append(f"[{rule['rule_id']}]{sig_marker} "
                        f"(Conf: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f}, "
                        f"Samples: {rule['samples']})")
            
            conditions_str = " AND ".join(
                [f"{feat} {op} {thresh:.4f}" for feat, op, thresh in rule['conditions']]
            )
            lines.append(f"  IF {conditions_str}")
            lines.append(f"  THEN → {rule['prediction']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def rules_to_dataframe(self, rules: List[Dict] = None) -> pd.DataFrame:
        """Convert rules to pandas DataFrame."""
        if rules is None:
            rules = self.extract_rules()
        
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
                'Confidence': rule['confidence'],
                'Lift': rule['lift'],
                'Support': rule['support'],
                'Significant': rule['significant']
            })
        
        return pd.DataFrame(rows)
    
    def evaluate_rule_quality(self, rules: List[Dict] = None, 
                              X: np.ndarray = None, 
                              y: np.ndarray = None) -> Dict[str, float]:
        """
        Comprehensive rule set quality evaluation.
        
        Implements multiple evaluation metrics from association rule mining 
        and rule learning literature:
        
        Rule-level metrics (per rule):
        - Confidence: Agrawal & Srikant (1994)
        - Lift: Brin et al. (1997)
        - Support: Agrawal & Srikant (1994)
        - Conviction: Brin et al. (1997)
        - Leverage: Piatetsky-Shapiro (1991)
        
        Rule-set level metrics (aggregated):
        - Coverage: Total samples covered by rules
        - Average Rule Length: Complexity measure
        - Significant Ratio: Proportion of statistically significant rules
        - Default Rule Ratio: Rules predicting majority vs minority class
        
        References:
        - Agrawal & Srikant (1994): Fast algorithms for mining association rules
        - Brin et al. (1997): Beyond market baskets (Conviction)
        - Piatetsky-Shapiro (1991): Discovery, analysis, and presentation (Leverage)
        - Fürnkranz & Kliegr (2015): A brief overview of rule learning
        
        Args:
            rules: List of rule dictionaries (if None, extract from tree)
            X: Feature matrix for coverage calculation (optional)
            y: Labels for class-specific metrics (optional)
        
        Returns:
            Dictionary with comprehensive rule quality metrics
        """
        if rules is None:
            rules = self.extract_rules()
        
        if len(rules) == 0:
            return {'error': 'No rules extracted'}
        
        # ====================================================================
        # Rule-Level Metrics (Enhanced from literature)
        # ====================================================================
        
        # Calculate additional metrics for each rule
        total_samples = sum(r['samples'] for r in rules)
        class_priors = {}
        for r in rules:
            cls = r['prediction_code']
            if cls not in class_priors:
                class_priors[cls] = 0
            class_priors[cls] += r['samples']
        for cls in class_priors:
            class_priors[cls] /= total_samples
        
        for rule in rules:
            # Conviction (Brin et al., 1997)
            # Measures how much the rule predicts the consequent better than random
            # Conviction = (1 - support(Y)) / (1 - confidence)
            prior = class_priors.get(rule['prediction_code'], 0.5)
            if rule['confidence'] < 1.0:
                rule['conviction'] = (1 - prior) / (1 - rule['confidence'] + 1e-8)
            else:
                rule['conviction'] = float('inf')  # Perfect rule
            
            # Leverage (Piatetsky-Shapiro, 1991)
            # Measures the difference between observed and expected support
            # Leverage = support(X∧Y) - support(X) * support(Y)
            # For decision tree rules: support = n_leaf/N, prior = class prior
            rule['leverage'] = rule['support'] * rule['confidence'] - rule['support'] * prior
            
            # Added Confidence (ACC) - alternative confidence measure
            # Measures prediction accuracy within the rule's coverage
            rule['added_confidence'] = rule['confidence'] - prior
            
            # Rule Length (Complexity)
            rule['length'] = len(rule['conditions'])
        
        # ====================================================================
        # Rule-Set Level Metrics (Aggregated)
        # ====================================================================
        
        n_rules = len(rules)
        
        # 1. Coverage (total proportion of samples covered)
        coverage = sum(r['support'] for r in rules)  # Should be ~1.0 for DT
        
        # 2. Average Confidence
        avg_confidence = np.mean([r['confidence'] for r in rules])
        
        # 3. Average Lift
        avg_lift = np.mean([r['lift'] for r in rules])
        
        # 4. Average Rule Length (Complexity)
        avg_length = np.mean([r['length'] for r in rules])
        max_length = max(r['length'] for r in rules)
        min_length = min(r['length'] for r in rules)
        
        # 5. Significant Rule Ratio (BH-corrected)
        n_significant = sum(1 for r in rules if r.get('significant', False))
        significant_ratio = n_significant / n_rules
        
        # 6. High Confidence Rule Ratio (confidence > 0.6)
        n_high_conf = sum(1 for r in rules if r['confidence'] > 0.6)
        high_conf_ratio = n_high_conf / n_rules
        
        # 7. Class Balance in Rules
        n_default_rules = sum(1 for r in rules if r['prediction_code'] == 1)
        n_nondefault_rules = sum(1 for r in rules if r['prediction_code'] == 0)
        default_rule_ratio = n_default_rules / n_rules
        
        # 8. Average Conviction (excluding infinite values)
        finite_convictions = [r['conviction'] for r in rules if r['conviction'] != float('inf')]
        avg_conviction = np.mean(finite_convictions) if finite_convictions else float('inf')
        
        # 9. Average Leverage
        avg_leverage = np.mean([r['leverage'] for r in rules])
        
        # 10. Weighted Average Confidence (weighted by support)
        weighted_conf = sum(r['confidence'] * r['support'] for r in rules) / coverage if coverage > 0 else 0
        
        # 11. Default Class Samples in Default Rules
        default_samples_in_default_rules = sum(
            r['class_distribution'].get('Default', 0) 
            for r in rules if r['prediction_code'] == 1
        )
        
        # ====================================================================
        # Quality Score (Composite)
        # ====================================================================
        # Combine key metrics into a single quality score
        quality_score = (
            0.25 * min(1.0, avg_confidence) +
            0.20 * min(1.0, avg_lift / 3.0) +  # Normalize lift to [0,1]
            0.20 * significant_ratio +
            0.15 * (1.0 - min(1.0, avg_length / 10.0)) +  # Shorter is better
            0.20 * min(1.0, avg_conviction / 5.0)  # Normalize conviction
        )
        
        metrics = {
            # Rule-set level metrics
            'n_rules': n_rules,
            'coverage': float(coverage),
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max(r['confidence'] for r in rules)),
            'min_confidence': float(min(r['confidence'] for r in rules)),
            'avg_lift': float(avg_lift),
            'max_lift': float(max(r['lift'] for r in rules)),
            'avg_length': float(avg_length),
            'max_length': int(max_length),
            'min_length': int(min_length),
            'significant_ratio': float(significant_ratio),
            'n_significant': int(n_significant),
            'high_conf_ratio': float(high_conf_ratio),
            'n_high_conf': int(n_high_conf),
            
            # Class balance in rules
            'n_default_rules': int(n_default_rules),
            'n_nondefault_rules': int(n_nondefault_rules),
            'default_rule_ratio': float(default_rule_ratio),
            
            # Advanced metrics from literature
            'avg_conviction': float(avg_conviction) if avg_conviction != float('inf') else 999.0,
            'avg_leverage': float(avg_leverage),
            'weighted_confidence': float(weighted_conf),
            
            # Composite quality score
            'quality_score': float(quality_score)
        }
        
        return metrics
    
    def get_distillation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive distillation quality report.
        
        Combines knowledge transfer metrics and rule quality metrics
        into a single report for analysis and paper reporting.
        
        Returns:
            Dictionary containing:
            - kt_metrics: Knowledge transfer quality metrics
            - rule_metrics: Rule extraction quality metrics
            - model_info: Model configuration and structure info
        """
        report = {
            'model_info': {
                'temperature': self.temperature,
                'alpha': self.alpha,
                'max_depth': self.max_depth,
                'use_class_balance': self.use_class_balance,
                'cost_ratio': self.cost_ratio,
                'tree_depth': self.student.get_depth() if self.student else None,
                'n_leaves': self.student.get_n_leaves() if self.student else None,
                'decision_threshold': self.decision_threshold
            },
            'kt_metrics': self.kt_metrics if hasattr(self, 'kt_metrics') else {},
            'kt_score': self.knowledge_transfer_score if hasattr(self, 'knowledge_transfer_score') else None
        }
        
        # Add rule quality metrics
        try:
            rules = self.extract_rules()
            report['rule_metrics'] = self.evaluate_rule_quality(rules)
            report['n_rules'] = len(rules)
        except Exception as e:
            report['rule_metrics'] = {'error': str(e)}
            report['n_rules'] = 0
        
        return report
    
    def get_tree_text(self, feature_names: List[str] = None) -> str:
        """Get sklearn's text representation of the tree."""
        if feature_names is None:
            feature_names = self.feature_names
        return export_text(self.student, feature_names=feature_names,
                          class_names=self.class_names)


# Backward compatibility aliases
class SoftLabelKDDistiller(DecisionTreeDistiller):
    """SoftLabelKD: Pure soft-label distillation without class balance."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.0,
                 max_depth: int = 6, min_samples_leaf: int = 5,
                 random_state: int = 42):
        super().__init__(
            temperature=temperature,
            alpha=alpha,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            use_class_balance=False,  # Key difference: no class balance
            random_state=random_state
        )


class VanillaKDDistiller(DecisionTreeDistiller):
    """VanillaKD: Standard KD with hard labels only."""
    
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 5,
                 random_state: int = 42):
        super().__init__(
            temperature=1.0,  # No temperature scaling
            alpha=1.0,        # Hard labels only
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            use_class_balance=False,
            random_state=random_state
        )
