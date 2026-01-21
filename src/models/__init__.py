#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Models module for credit scoring.

Modules:
    - baselines: Traditional ML models (LR, SVM, RF, XGBoost, LightGBM, CatBoost)
    - neural: CreditNet neural network architecture (optimized for credit scoring)
"""

from .baselines import get_baseline_models, ModelConfig
from .neural import CreditNet, NeuralNetworkTrainer, create_teacher_model, train_all_teacher_models

__all__ = [
    'get_baseline_models',
    'ModelConfig',
    'CreditNet',
    'NeuralNetworkTrainer',
    'create_teacher_model',
    'train_all_teacher_models'
]
