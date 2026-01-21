#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Loss Functions

Implemented methods:
    - VanillaKDLoss: Hinton et al., 2015
    - FitNetsLoss: Romero et al., 2015
    - AttentionLoss: Zagoruyko & Komodakis, 2017
    - SAKDLoss: SHAP-guided Adaptive KD (Ours)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class VanillaKDLoss(nn.Module):
    """
    Vanilla Knowledge Distillation.
    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    Supports both binary (CreditNet) and multi-class (TabularMLP) outputs.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
    
    def _is_binary_output(self, logits: torch.Tensor) -> bool:
        """Check if output is binary (single value per sample)."""
        return logits.shape[-1] == 1 or len(logits.shape) == 1
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor, 
                labels: torch.Tensor, **kwargs) -> torch.Tensor:
        
        is_binary = self._is_binary_output(student_logits)
        
        if is_binary:
            # Binary classification (CreditNet style)
            s_logits = student_logits.squeeze()
            t_logits = teacher_logits.squeeze()
            labels_float = labels.float()
            
            # Hard label loss using BCE
            s_probs = torch.sigmoid(s_logits) if s_logits.max() > 1 or s_logits.min() < 0 else s_logits
            ce_loss = self.bce_loss(s_probs, labels_float)
            
            # Soft label loss (temperature-scaled)
            t_soft = torch.sigmoid(t_logits / self.temperature)
            s_soft = torch.sigmoid(s_logits / self.temperature)
            kd_loss = F.mse_loss(s_soft, t_soft) * (self.temperature ** 2)
        else:
            # Multi-class classification (TabularMLP style)
            # Hard label loss
            ce_loss = self.ce_loss(student_logits, labels)
            
            # Soft label loss
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
            kd_loss = kd_loss * (self.temperature ** 2)
        
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss


class FitNetsLoss(nn.Module):
    """
    FitNets: Hints for Thin Deep Nets.
    Reference: Romero et al., ICLR 2015
    Supports both binary (CreditNet) and multi-class (TabularMLP) outputs.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, 
                 hint_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hint_weight = hint_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.regressor = None
    
    def _is_binary_output(self, logits: torch.Tensor) -> bool:
        return logits.shape[-1] == 1 or len(logits.shape) == 1
    
    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor,
                student_features: torch.Tensor = None,
                teacher_features: torch.Tensor = None, **kwargs) -> torch.Tensor:
        
        is_binary = self._is_binary_output(student_logits)
        
        if is_binary:
            # Binary classification
            s_logits = student_logits.squeeze()
            t_logits = teacher_logits.squeeze()
            labels_float = labels.float()
            
            s_probs = torch.sigmoid(s_logits) if s_logits.max() > 1 or s_logits.min() < 0 else s_logits
            ce_loss = self.bce_loss(s_probs, labels_float)
            
            t_soft = torch.sigmoid(t_logits / self.temperature)
            s_soft = torch.sigmoid(s_logits / self.temperature)
            kd_loss = F.mse_loss(s_soft, t_soft) * (self.temperature ** 2)
        else:
            # Multi-class classification
            ce_loss = self.ce_loss(student_logits, labels)
            
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
            kd_loss = kd_loss * (self.temperature ** 2)
        
        # Hint loss (same for both)
        hint_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            if student_features.shape[1] != teacher_features.shape[1]:
                if self.regressor is None or \
                   self.regressor.in_features != student_features.shape[1]:
                    self.regressor = nn.Linear(
                        student_features.shape[1], 
                        teacher_features.shape[1]
                    ).to(student_logits.device)
                student_features = self.regressor(student_features)
            hint_loss = self.mse_loss(student_features, teacher_features)
        
        total = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return total + self.hint_weight * hint_loss


class AttentionLoss(nn.Module):
    """
    Attention Transfer.
    Reference: Zagoruyko & Komodakis, ICLR 2017
    Supports both binary (CreditNet) and multi-class (TabularMLP) outputs.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, 
                 beta: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
    
    def _is_binary_output(self, logits: torch.Tensor) -> bool:
        return logits.shape[-1] == 1 or len(logits.shape) == 1
    
    def attention_map(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features.pow(2).mean(dim=1), p=2, dim=0)
    
    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor,
                student_features: torch.Tensor = None,
                teacher_features: torch.Tensor = None, **kwargs) -> torch.Tensor:
        
        is_binary = self._is_binary_output(student_logits)
        
        if is_binary:
            # Binary classification
            s_logits = student_logits.squeeze()
            t_logits = teacher_logits.squeeze()
            labels_float = labels.float()
            
            s_probs = torch.sigmoid(s_logits) if s_logits.max() > 1 or s_logits.min() < 0 else s_logits
            ce_loss = self.bce_loss(s_probs, labels_float)
            
            t_soft = torch.sigmoid(t_logits / self.temperature)
            s_soft = torch.sigmoid(s_logits / self.temperature)
            kd_loss = F.mse_loss(s_soft, t_soft) * (self.temperature ** 2)
        else:
            # Multi-class classification
            ce_loss = self.ce_loss(student_logits, labels)
            
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
            kd_loss = kd_loss * (self.temperature ** 2)
        
        # Attention loss (same for both)
        att_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            s_att = self.attention_map(student_features)
            t_att = self.attention_map(teacher_features)
            att_loss = (s_att - t_att).pow(2).mean()
        
        total = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return total + self.beta * att_loss


class SAKDLoss(nn.Module):
    """
    SHAP-guided Adaptive Knowledge Distillation (Ours).
    
    Key innovations:
        1. SHAP-guided feature importance alignment
        2. Adaptive temperature scaling based on teacher confidence
        3. Interpretability-preserving regularization
        4. Multi-granularity knowledge transfer
    
    Supports both binary (CreditNet) and multi-class (TabularMLP) outputs.
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.3,
                 beta: float = 0.3,
                 shap_weight: float = 0.2,
                 interp_weight: float = 0.2,
                 adaptive_temp: bool = False):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # CE weight
        self.beta = beta    # KD weight
        self.shap_weight = shap_weight  # SHAP alignment weight
        self.interp_weight = interp_weight  # Interpretability weight
        self.adaptive_temp = adaptive_temp
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.shap_importance = None
    
    def _is_binary_output(self, logits: torch.Tensor) -> bool:
        return logits.shape[-1] == 1 or len(logits.shape) == 1
    
    def set_shap_importance(self, shap_values: np.ndarray):
        """Set SHAP-based feature importance."""
        shap_arr = np.array(shap_values)
        
        # Handle multi-dimensional SHAP
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1] if shap_arr.shape[2] > 1 else shap_arr[:, :, 0]
        
        # Mean absolute importance
        importance = np.abs(shap_arr).mean(axis=0).flatten()
        importance = importance / (importance.sum() + 1e-8)
        
        self.shap_importance = torch.from_numpy(importance).float()
    
    def get_adaptive_temperature(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute adaptive temperature based on teacher confidence."""
        if not self.adaptive_temp:
            return self.temperature
        
        probs = F.softmax(teacher_logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # Lower confidence -> higher temperature
        adaptive_tau = self.temperature * (1 + 0.5 * (1 - confidence))
        return adaptive_tau.unsqueeze(1)
    
    def shap_alignment_loss(self, student_importance: torch.Tensor) -> torch.Tensor:
        """Compute SHAP importance alignment loss."""
        if self.shap_importance is None:
            return torch.tensor(0.0, device=student_importance.device)
        
        target = self.shap_importance.to(student_importance.device)
        return self.mse_loss(student_importance, target)
    
    def interpretability_loss(self, student_importance: torch.Tensor) -> torch.Tensor:
        """Compute interpretability regularization (sparsity + entropy)."""
        # L1 sparsity
        sparsity = student_importance.abs().sum()
        
        # Entropy (encourage peaked distribution)
        entropy = -(student_importance * torch.log(student_importance + 1e-8)).sum()
        
        return 0.5 * sparsity + 0.5 * entropy
    
    def get_adaptive_temperature_binary(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute adaptive temperature for binary classification."""
        if not self.adaptive_temp:
            return self.temperature
        
        t_logits = teacher_logits.squeeze()
        probs = torch.sigmoid(t_logits)
        confidence = torch.max(probs, 1 - probs)
        
        # Lower confidence -> higher temperature
        adaptive_tau = self.temperature * (1 + 0.5 * (1 - confidence))
        return adaptive_tau
    
    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor,
                student_model: nn.Module = None, **kwargs) -> torch.Tensor:
        
        is_binary = self._is_binary_output(student_logits)
        
        if is_binary:
            # Binary classification (CreditNet style)
            s_logits = student_logits.squeeze()
            t_logits = teacher_logits.squeeze()
            labels_float = labels.float()
            
            # 1. Hard label loss using BCE
            s_probs = torch.sigmoid(s_logits) if s_logits.max() > 1 or s_logits.min() < 0 else s_logits
            ce_loss = self.bce_loss(s_probs, labels_float)
            
            # 2. Knowledge distillation loss
            if self.adaptive_temp:
                tau = self.get_adaptive_temperature_binary(teacher_logits)
                t_soft = torch.sigmoid(t_logits / tau)
                s_soft = torch.sigmoid(s_logits / tau)
                scale = (tau ** 2).mean()
            else:
                t_soft = torch.sigmoid(t_logits / self.temperature)
                s_soft = torch.sigmoid(s_logits / self.temperature)
                scale = self.temperature ** 2
            
            kd_loss = F.mse_loss(s_soft, t_soft) * scale
        else:
            # Multi-class classification (TabularMLP style)
            # 1. Hard label loss
            ce_loss = self.ce_loss(student_logits, labels)
            
            # 2. Knowledge distillation loss
            if self.adaptive_temp:
                tau = self.get_adaptive_temperature(teacher_logits)
                soft_targets = F.softmax(teacher_logits / tau, dim=1)
                soft_student = F.log_softmax(student_logits / tau, dim=1)
                scale = (tau ** 2).mean()
            else:
                soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
                scale = self.temperature ** 2
            
            kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * scale
        
        # 3. SHAP alignment loss
        shap_loss = torch.tensor(0.0, device=student_logits.device)
        if student_model is not None and hasattr(student_model, 'feature_importance'):
            student_imp = F.softmax(student_model.feature_importance, dim=0)
            shap_loss = self.shap_alignment_loss(student_imp)
        
        # 4. Interpretability loss
        interp_loss = torch.tensor(0.0, device=student_logits.device)
        if student_model is not None and hasattr(student_model, 'feature_importance'):
            student_imp = F.softmax(student_model.feature_importance, dim=0)
            interp_loss = self.interpretability_loss(student_imp)
        
        # Total loss
        total = (self.alpha * ce_loss + 
                 self.beta * kd_loss + 
                 self.shap_weight * shap_loss + 
                 self.interp_weight * interp_loss)
        
        return total
