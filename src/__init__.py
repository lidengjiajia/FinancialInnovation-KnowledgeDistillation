"""
SAKD: SHAP-guided Adaptive Knowledge Distillation for Credit Scoring
"""

__version__ = "1.0.0"

from .data import DataPreprocessor, CreditDataset
from .models import get_baseline_models, CreditNet, NeuralNetworkTrainer, create_teacher_model
from .distillation import SAKDLoss, VanillaKDLoss, FitNetsLoss, AttentionLoss, DistillationTrainer
