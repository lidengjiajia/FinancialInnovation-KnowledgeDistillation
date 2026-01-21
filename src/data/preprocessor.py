#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preprocessing Module for Credit Scoring

Datasets:
    - German Credit (UCI)
    - Australian Credit (UCI)
    - Xinwang Credit (Chinese P2P)
    - UCI Credit Card Default
"""

import os
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(use_gpu: bool = True) -> torch.device:
    """Get computing device."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[INFO] Using CPU")
    
    return device


class CreditDataset(Dataset):
    """
    PyTorch Dataset for credit scoring data.
    
    Args:
        X: Feature matrix
        y: Target labels
        device: Computing device
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.X = torch.FloatTensor(X).to(self.device)
        self.y = torch.LongTensor(y).to(self.device)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# =============================================================================
# Data Preprocessor Class
# =============================================================================

class DataPreprocessor:
    """
    Comprehensive data preprocessor for credit scoring datasets.
    
    Features:
        - Multiple dataset support
        - Automatic preprocessing
        - Feature scaling
        - Train/val/test splitting
        
    Attributes:
        scalers: Dictionary of fitted scalers
        encoders: Dictionary of fitted label encoders
        feature_names: Dictionary of feature names per dataset
    """
    
    def __init__(self, data_dir: str = 'data', random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Directory containing data files
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Dict[str, List[str]] = {}
        
        set_random_seed(random_state)
    
    # -------------------------------------------------------------------------
    # German Credit Dataset
    # -------------------------------------------------------------------------
    
    def load_german_credit(self) -> Optional[Dict]:
        """
        Load and preprocess German Credit dataset.
        
        Returns:
            Dictionary containing processed data or None if file not found
        """
        print("[INFO] Loading German Credit dataset...")
        
        filepath = os.path.join(self.data_dir, 'german_credit.csv')
        
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"  Original shape: {df.shape}")
        
        # Identify target column
        target_col = self._find_target_column(df, ['class', 'Class', 'target'])
        
        # Convert target to binary (1=good, 0=bad)
        if df[target_col].nunique() == 2:
            unique_vals = sorted(df[target_col].unique())
            # German dataset: 1=good, 2=bad -> 1=good, 0=bad
            if 2 in unique_vals:
                df[target_col] = df[target_col].replace({1: 1, 2: 0})
        
        # Identify column types
        categorical_cols, numerical_cols = self._identify_column_types(
            df, target_col, max_unique=10
        )
        
        # One-hot encode categorical variables
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col].values
        
        # Store feature names
        self.feature_names['german'] = list(X.columns)
        
        # Split and scale data
        return self._split_and_scale(X.values, y, 'german')
    
    # -------------------------------------------------------------------------
    # Australian Credit Dataset
    # -------------------------------------------------------------------------
    
    def load_australian_credit(self) -> Optional[Dict]:
        """
        Load and preprocess Australian Credit dataset.
        
        Returns:
            Dictionary containing processed data or None if file not found
        """
        print("[INFO] Loading Australian Credit dataset...")
        
        filepath = os.path.join(self.data_dir, 'australian_credit.csv')
        
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"  Original shape: {df.shape}")
        
        # Identify target column
        target_col = self._find_target_column(df, ['Class', 'class', 'A15'])
        
        # Ensure binary target
        unique_vals = df[target_col].unique()
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[target_col] = df[target_col].map(mapping)
        
        # Handle missing values
        df = self._handle_missing_values(df, target_col)
        
        # Identify column types
        categorical_cols, numerical_cols = self._identify_column_types(
            df, target_col, max_unique=10
        )
        
        # One-hot encode categorical variables
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col].values
        
        # Store feature names
        self.feature_names['australian'] = list(X.columns)
        
        return self._split_and_scale(X.values, y, 'australian')
    
    # -------------------------------------------------------------------------
    # Xinwang Credit Dataset
    # -------------------------------------------------------------------------
    
    def load_xinwang_credit(self) -> Optional[Dict]:
        """
        Load and preprocess Xinwang Credit dataset (Chinese P2P lending).
        
        Returns:
            Dictionary containing processed data or None if file not found
        """
        print("[INFO] Loading Xinwang Credit dataset...")
        
        filepath = os.path.join(self.data_dir, 'xinwang.csv')
        
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"  Original shape: {df.shape}")
        
        target_col = 'target'
        
        # Handle missing values encoded as -99
        for col in df.columns:
            if col != target_col:
                df[col] = df[col].replace(-99, np.nan)
                if df[col].dtype != 'object':
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_val = df[col].mode()
                    df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
        
        # Encode categorical columns
        categorical_cols = ['province', 'industry', 'scope', 'judicial']
        for col in categorical_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[f'xinwang_{col}'] = le
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Ensure binary target
        if y.min() != 0 or y.max() != 1:
            y = (y > 0).astype(int)
        
        # Store feature names
        self.feature_names['xinwang'] = list(X.columns)
        
        return self._split_and_scale(X.values, y, 'xinwang')
    
    # -------------------------------------------------------------------------
    # UCI Credit Card Default Dataset
    # -------------------------------------------------------------------------
    
    def load_uci_credit(self) -> Optional[Dict]:
        """
        Load and preprocess UCI Credit Card Default dataset.
        
        Returns:
            Dictionary containing processed data or None if file not found
        """
        print("[INFO] Loading UCI Credit Card Default dataset...")
        
        filepath = os.path.join(self.data_dir, 'uci_credit.xls')
        
        if not os.path.exists(filepath):
            # Try CSV format
            filepath = os.path.join(self.data_dir, 'uci_credit.csv')
            if not os.path.exists(filepath):
                print(f"[ERROR] File not found: {filepath}")
                return None
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, header=1, index_col=0)
        
        print(f"  Original shape: {df.shape}")
        
        # Standardize column names
        if 'default payment next month' in df.columns:
            df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
        
        target_col = 'DEFAULT'
        
        # Remove ID column if exists
        if 'ID' in df.columns:
            df.drop('ID', axis=1, inplace=True)
        
        # Handle anomalous values
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].replace({0: 2})
        if 'EDUCATION' in df.columns:
            df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
        if 'MARRIAGE' in df.columns:
            df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
        
        # Force numeric conversion
        for col in df.columns:
            if col != target_col:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col].astype(int).values
        
        # Store feature names
        self.feature_names['uci'] = list(X.columns)
        
        return self._split_and_scale(X.values, y, 'uci')

    # -------------------------------------------------------------------------
    # Compatibility aliases (for the updated module interface)
    # -------------------------------------------------------------------------

    def load_xinwang(self) -> Optional[Dict]:
        """Alias of load_xinwang_credit()."""
        return self.load_xinwang_credit()

    def process_all_datasets(self) -> Dict[str, Dict]:
        """Process all supported datasets and return a unified dict.

        This mirrors the interface used by the NN/SHAP plotting utilities.
        Keys: german / australian / uci / xinwang
        """
        processed_data: Dict[str, Dict] = {}

        loaders = {
            'german': self.load_german_credit,
            'australian': self.load_australian_credit,
            'uci': self.load_uci_credit,
            'xinwang': self.load_xinwang_credit,
        }

        for name, fn in loaders.items():
            data = fn()
            if data is not None:
                processed_data[name] = data

        return processed_data
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _find_target_column(self, df: pd.DataFrame, 
                            candidates: List[str]) -> str:
        """Find target column from candidates or use last column."""
        for col in candidates:
            if col in df.columns:
                return col
        return df.columns[-1]
    
    def _identify_column_types(self, df: pd.DataFrame, target_col: str,
                               max_unique: int = 10) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical columns."""
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if col != target_col:
                if df[col].dtype == 'object' or df[col].nunique() <= max_unique:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        return categorical_cols, numerical_cols
    
    def _handle_missing_values(self, df: pd.DataFrame, 
                               target_col: str) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        for col in df.columns:
            if col != target_col:
                if df[col].dtype == 'object':
                    mode_val = df[col].mode()
                    df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        return df
    
    def _split_and_scale(self, X: np.ndarray, y: np.ndarray,
                         dataset_name: str,
                         test_size: float = 0.2,
                         val_size: float = 0.2) -> Dict:
        """
        Split data into train/val/test and apply standardization.
        
        Args:
            X: Feature matrix
            y: Target labels
            dataset_name: Name for storing scaler
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Dictionary with processed data
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Handle NaN/Inf values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store scaler
        self.scalers[dataset_name] = scaler
        
        print(f"  Processed shape: Features={X_train.shape[1]}")
        print(f"  Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        print(f"  Class distribution (train): {np.bincount(y_train)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names.get(dataset_name, []),
            'scaler': scaler
        }
    
    # -------------------------------------------------------------------------
    # Main Loading Function
    # -------------------------------------------------------------------------
    
    def load_all_datasets(self) -> Dict[str, Dict]:
        """
        Load and preprocess all available datasets.
        
        Returns:
            Dictionary mapping dataset names to processed data
        """
        print("=" * 60)
        print("Loading All Credit Scoring Datasets")
        print("=" * 60)
        
        datasets = {}
        
        # German Credit
        german_data = self.load_german_credit()
        if german_data:
            datasets['german'] = german_data
        
        # Australian Credit
        australian_data = self.load_australian_credit()
        if australian_data:
            datasets['australian'] = australian_data
        
        # Xinwang Credit
        xinwang_data = self.load_xinwang_credit()
        if xinwang_data:
            datasets['xinwang'] = xinwang_data
        
        # UCI Credit
        uci_data = self.load_uci_credit()
        if uci_data:
            datasets['uci'] = uci_data
        
        print("=" * 60)
        print(f"Successfully loaded {len(datasets)} datasets")
        print("=" * 60)
        
        return datasets


# =============================================================================
# Convenience Functions
# =============================================================================

def load_dataset(name: str, data_dir: str = 'data') -> Optional[Dict]:
    """
    Load a single dataset by name.
    
    Args:
        name: Dataset name ('german', 'australian', 'xinwang', 'uci')
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with processed data or None
    """
    preprocessor = DataPreprocessor(data_dir=data_dir)
    
    loaders = {
        'german': preprocessor.load_german_credit,
        'australian': preprocessor.load_australian_credit,
        'xinwang': preprocessor.load_xinwang_credit,
        'uci': preprocessor.load_uci_credit
    }
    
    if name not in loaders:
        print(f"[ERROR] Unknown dataset: {name}")
        print(f"Available datasets: {list(loaders.keys())}")
        return None
    
    return loaders[name]()


def create_dataloaders(data: Dict, batch_size: int = 128,
                       device: torch.device = None) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from preprocessed data.
    
    Args:
        data: Dictionary with X_train, y_train, etc.
        batch_size: Batch size for DataLoader
        device: Computing device
        
    Returns:
        Dictionary with train, val, test DataLoaders
    """
    device = device or get_device()
    
    train_dataset = CreditDataset(data['X_train'], data['y_train'], device)
    val_dataset = CreditDataset(data['X_val'], data['y_val'], device)
    test_dataset = CreditDataset(data['X_test'], data['y_test'], device)
    
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }



