#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distillation Training Module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """Training manager for knowledge distillation."""
    
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 loss_fn: nn.Module,
                 device: torch.device = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.teacher = teacher_model
        self.student = student_model
        self.loss_fn = loss_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.teacher.eval()
        
        self.optimizer = optim.Adam(
            self.student.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        
        for X, y in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            with torch.no_grad():
                t_logits = self.teacher(X)
                if hasattr(self.teacher, 'hidden_features'):
                    t_features = self.teacher.hidden_features
                else:
                    t_features = None
            
            s_logits = self.student(X)
            if hasattr(self.student, 'hidden_features'):
                s_features = self.student.hidden_features
            else:
                s_features = None
            
            loss = self.loss_fn(
                student_logits=s_logits,
                teacher_logits=t_logits,
                labels=y,
                student_features=s_features,
                teacher_features=t_features,
                student_model=self.student
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader: DataLoader, dataset_name: str = None) -> Dict[str, float]:
        """Evaluate student model."""
        self.student.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Check if model outputs binary (CreditNet) or multi-class (TabularMLP)
        # CreditNet outputs shape: (batch, 1) with sigmoid/logits
        # TabularMLP outputs shape: (batch, 2) with logits for 2 classes
        
        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                outputs = self.student(X)
                
                # Determine output format
                if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                    # Binary output (CreditNet style)
                    outputs = outputs.squeeze()
                    # Check if output is logits or already sigmoid
                    if hasattr(self.student, 'sigmoid') and isinstance(self.student.sigmoid, nn.Identity):
                        # Output is logits, apply sigmoid
                        probs = torch.sigmoid(outputs)
                    else:
                        # Output is already sigmoid probabilities
                        probs = outputs
                    preds = (probs > 0.5).float()
                else:
                    # Multi-class output (TabularMLP style) 
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                                     precision_score, recall_score)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              early_stop: int = 15) -> Dict[str, float]:
        """Full training loop with early stopping."""
        best_auc = 0.0
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(train_loss)
            
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_state = self.student.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}: loss={train_loss:.4f}, "
                           f"val_auc={val_metrics['auc']:.4f}")
            
            if no_improve >= early_stop:
                logger.info(f"Early stop at epoch {epoch+1}")
                break
        
        if best_state:
            self.student.load_state_dict(best_state)
        
        return self.evaluate(val_loader)
