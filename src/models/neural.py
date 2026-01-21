#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经网络模型模块 - 更新版
Neural Network Models Module - Updated Version
使用PyTorch实现的信用评分神经网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果使用多GPU
# 确保PyTorch算法确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 并发配置：Windows平台使用较少的worker避免问题
import platform
if platform.system() == 'Windows':
    # Windows上设置为0避免multiprocessing问题
    n_workers = 0
else:
    # Linux/Mac可以使用更多worker
    n_workers = max(1, min(os.cpu_count() - 1, os.cpu_count()))
# 只在需要时显示配置信息，避免重复输出

# 只在第一次导入时显示设备信息
_device_shown = False

class CreditNet(nn.Module):
    """Advanced Credit Scoring Neural Network
    
    Based on recent research in credit scoring neural networks:
    - arXiv:2411.17783: Kolmogorov-Arnold Networks for Credit Default Prediction  
    - arXiv:2412.02097: Hybrid Model of KAN and gMLP for Large-Scale Financial Data
    - arXiv:2209.10070: Monotonic Neural Additive Models for Credit Scoring
    """
    
    def __init__(self, input_dim, dataset_type='german'):
        super(CreditNet, self).__init__()
        
        # Advanced architectures based on recent research
        if dataset_type == 'german':
            # 改进的German数据集架构 - 针对不平衡数据优化
            # 使用Residual连接和更深的网络，参考信用评分研究中的最佳实践
            # Target: 提高准确率到75%+
            self.input_layer = nn.Linear(input_dim, 512)
            self.bn1 = nn.BatchNorm1d(512)
            
            # 第一个残差块
            self.fc1 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 256)
            self.bn3 = nn.BatchNorm1d(256)
            self.shortcut1 = nn.Linear(512, 256)  # shortcut connection
            
            # 第二个残差块
            self.fc3 = nn.Linear(256, 128)
            self.bn4 = nn.BatchNorm1d(128)
            self.fc4 = nn.Linear(128, 128)
            self.bn5 = nn.BatchNorm1d(128)
            self.shortcut2 = nn.Linear(256, 128)  # shortcut connection
            
            # 最终分类层
            self.fc5 = nn.Linear(128, 64)
            self.bn6 = nn.BatchNorm1d(64)
            self.fc6 = nn.Linear(64, 32)
            self.fc7 = nn.Linear(32, 1)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.sigmoid = nn.Sigmoid()
            
        elif dataset_type == 'australian':
            # Medium complexity for balanced dataset
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128), 
                nn.ReLU(),
                nn.Dropout(0.35),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.25),
                
                nn.Linear(32, 1)
            )
            self.sigmoid = nn.Sigmoid()  # Separate sigmoid for logits mode support
        elif dataset_type == 'uci':
            # Deep architecture for large dataset (30k samples)
            # Inspired by arXiv:2412.02097 hybrid approaches
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.45),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.35),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            self.sigmoid = nn.Sigmoid()  # Separate sigmoid for logits mode support
        elif dataset_type == 'xinwang':
            # Very deep architecture for large Xinwang dataset (17,886 samples, 97 features)
            # Designed for high-dimensional credit risk assessment
            # Uses residual connections to enable deeper networks
            self.input_layer = nn.Linear(input_dim, 512)
            self.bn1 = nn.BatchNorm1d(512)
            
            # 第一个残差块
            self.fc1 = nn.Linear(512, 384)
            self.bn2 = nn.BatchNorm1d(384)
            self.fc2 = nn.Linear(384, 384)
            self.bn3 = nn.BatchNorm1d(384)
            self.shortcut1 = nn.Linear(512, 384)
            
            # 第二个残差块
            self.fc3 = nn.Linear(384, 256)
            self.bn4 = nn.BatchNorm1d(256)
            self.fc4 = nn.Linear(256, 256)
            self.bn5 = nn.BatchNorm1d(256)
            self.shortcut2 = nn.Linear(384, 256)
            
            # 第三个残差块
            self.fc5 = nn.Linear(256, 128)
            self.bn6 = nn.BatchNorm1d(128)
            self.fc6 = nn.Linear(128, 128)
            self.bn7 = nn.BatchNorm1d(128)
            self.shortcut3 = nn.Linear(256, 128)
            
            # 最终分类层
            self.fc7 = nn.Linear(128, 64)
            self.bn8 = nn.BatchNorm1d(64)
            self.fc8 = nn.Linear(64, 32)
            self.fc9 = nn.Linear(32, 1)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.4)
            self.sigmoid = nn.Sigmoid()
        
        # Weight initialization based on Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if hasattr(self, 'layers'):
            # 对于Australian和UCI数据集使用Sequential layers
            x = self.layers(x)
            # 应用sigmoid（可以被替换为Identity用于logits模式）
            if hasattr(self, 'sigmoid') and not isinstance(self.sigmoid, nn.Identity):
                x = self.sigmoid(x)
            return x
        else:
            # 对于German和Xinwang数据集使用残差连接
            # 输入层
            x = self.relu(self.bn1(self.input_layer(x)))
            x = self.dropout(x)
            
            # 第一个残差块
            identity1 = self.shortcut1(x)
            x = self.relu(self.bn2(self.fc1(x)))
            x = self.dropout(x)
            x = self.bn3(self.fc2(x))
            x = self.relu(x + identity1)  # 残差连接
            x = self.dropout(x)
            
            # 第二个残差块
            identity2 = self.shortcut2(x)
            x = self.relu(self.bn4(self.fc3(x)))
            x = self.dropout(x)
            x = self.bn5(self.fc4(x))
            x = self.relu(x + identity2)  # 残差连接
            x = self.dropout(x)
            
            # 第三个残差块（仅Xinwang使用）
            if hasattr(self, 'shortcut3'):
                identity3 = self.shortcut3(x)
                x = self.relu(self.bn6(self.fc5(x)))
                x = self.dropout(x)
                x = self.bn7(self.fc6(x))
                x = self.relu(x + identity3)  # 残差连接
                x = self.dropout(x)
                
                # Xinwang的最终分类层
                x = self.relu(self.bn8(self.fc7(x)))
                x = self.dropout(x)
                x = self.relu(self.fc8(x))
                x = self.fc9(x)
            else:
                # German的最终分类层
                x = self.relu(self.bn6(self.fc5(x)))
                x = self.dropout(x)
                x = self.relu(self.fc6(x))
                x = self.fc7(x)
            
            # 最后一层输出 - 支持logits或sigmoid
            if hasattr(self, 'sigmoid') and not isinstance(self.sigmoid, nn.Identity):
                x = self.sigmoid(x)
            
            return x

class NeuralNetworkTrainer:
    """神经网络训练器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model_advanced(self, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=150, patience=15, use_logits=False):
        """Advanced model training with learning rate scheduling and improved techniques"""
        best_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 数据验证 - 检查是否有NaN或Inf
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"⚠️  Warning: NaN or Inf detected in inputs, skipping batch")
                    continue
                
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    print(f"⚠️  Warning: NaN or Inf detected in labels, skipping batch")
                    continue
                
                # Forward pass
                outputs = model(inputs)
                
                # 检查模型输出
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"⚠️  Warning: NaN or Inf detected in model outputs")
                    print(f"   Outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                    continue
                
                # 根据损失函数类型调整标签格式
                if use_logits and isinstance(criterion, nn.BCEWithLogitsLoss):
                    # BCEWithLogitsLoss需要float标签(已经是float)，需要squeeze以匹配outputs形状
                    loss = criterion(outputs.squeeze(), labels)
                else:
                    # BCELoss需要unsqueeze的标签
                    loss = criterion(outputs, labels.unsqueeze(1))
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Warning: NaN or Inf loss detected")
                    print(f"   Outputs: {outputs.squeeze()[:5]}")
                    print(f"   Labels: {labels.float()[:5]}")
                    continue
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    
                    # 根据输出类型计算预测
                    if use_logits:
                        # 对于logits输出，使用sigmoid后判断
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                    else:
                        # 对于sigmoid输出，直接判断
                        preds = (outputs > 0.5).float()
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            val_acc = accuracy_score(val_labels, val_preds)
            
            train_losses.append(epoch_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping mechanism
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'     Early stopping at epoch {epoch+1} (best val acc: {best_acc:.4f})')
                break
            
            if (epoch + 1) % 30 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'     Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
        
        # Load best model state
        model.load_state_dict(best_model_state)
        return model, train_losses, val_accuracies, best_acc
        
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
        """Original training method (kept for compatibility)"""
        best_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            val_acc = accuracy_score(val_labels, val_preds)
            
            train_losses.append(epoch_loss)
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'     Early stopping at epoch {epoch+1}')
                break
            
            if (epoch + 1) % 20 == 0:
                print(f'     Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Load best model state
        model.load_state_dict(best_model_state)
        return model, train_losses, val_accuracies, best_acc
    
    def test_model(self, model, test_loader, use_logits=False):
        """测试模型"""
        model.eval()
        test_preds = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                
                # 根据输出类型计算预测和概率
                if use_logits:
                    # 对于logits输出
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                else:
                    # 对于sigmoid输出
                    probs = outputs
                    preds = (outputs > 0.5).float()
                
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
        
        # 计算评估指标
        y_true = np.asarray(test_labels).astype(int)
        y_pred = np.asarray(test_preds).astype(int).reshape(-1)
        y_prob = np.asarray(test_probs).astype(float).reshape(-1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': test_preds,
            'probabilities': test_probs,
            'true_labels': test_labels
        }
    
    def create_data_loaders(self, data_dict, batch_size=32):
        """创建数据加载器"""
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(data_dict['X_train'])
        # 确保标签是float类型以兼容BCEWithLogitsLoss
        y_train_tensor = torch.FloatTensor(data_dict['y_train'])
        X_val_tensor = torch.FloatTensor(data_dict['X_val'])
        y_val_tensor = torch.FloatTensor(data_dict['y_val'])
        X_test_tensor = torch.FloatTensor(data_dict['X_test'])
        y_test_tensor = torch.FloatTensor(data_dict['y_test'])
        
        # 创建数据加载器 - 使用并发加载
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=n_workers, pin_memory=True if device.type == 'cuda' else False)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=n_workers, pin_memory=True if device.type == 'cuda' else False)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=n_workers, pin_memory=True if device.type == 'cuda' else False)
        
        return train_loader, val_loader, test_loader

def create_teacher_model(dataset_name, processed_data):
    """Create and train teacher model with advanced optimization techniques
    
    Based on recent research findings:
    - Learning rate scheduling for better convergence
    - Class balancing for improved performance
    - Advanced optimization strategies
    """
    # Get data
    data_dict = processed_data[dataset_name]
    input_dim = data_dict['X_train'].shape[1]
    
    # Create trainer
    trainer = NeuralNetworkTrainer(device)
    
    # Dataset-specific hyperparameters based on research best practices
    if dataset_name == 'uci':
        batch_size = 128  # Larger batch for large dataset
        learning_rate = 0.001
        num_epochs = 200  # 减少epochs，通过early stopping确保收敛
        patience = 20
        weight_decay = 1e-4
    elif dataset_name == 'australian':
        batch_size = 64   # Medium batch for medium dataset
        learning_rate = 0.002
        num_epochs = 150  # 减少epochs
        patience = 15
        weight_decay = 1e-3
    elif dataset_name == 'xinwang':
        batch_size = 64   # 较大batch size适合数据量较大的数据集
        learning_rate = 0.001  # 适中学习率
        num_epochs = 150  # 适度epochs
        patience = 20     # 早停耐心
        weight_decay = 1e-3  # 正则化防止过拟合
    else:  # german - 优化的参数以提高性能并减少训练时间
        batch_size = 64   # 增大batch size以加速训练
        learning_rate = 0.001  # 适中的学习率平衡速度和稳定性
        num_epochs = 150  # 减少epochs但通过early stopping确保充分收敛
        patience = 20     # 适度耐心
        weight_decay = 1e-3  # 适度正则化防止过拟合
    
    train_loader, val_loader, test_loader = trainer.create_data_loaders(data_dict, batch_size)
    
    # Create model
    model = CreditNet(input_dim, dataset_name).to(device)
    
    # 为German和Xinwang数据集计算类别权重以处理不平衡数据
    if dataset_name in ['german', 'xinwang']:
        y_train = data_dict['y_train']
        # 计算类别权重：少数类权重更高
        n_samples = len(y_train)
        n_classes = len(np.unique(y_train))
        class_counts = np.bincount(y_train)
        # 使用balanced策略计算权重
        class_weights = n_samples / (n_classes * class_counts)
        print(f"     {dataset_name.upper()}数据集类别分布: {class_counts}")
        print(f"     计算的类别权重: {class_weights}")
        
        # 将权重转换为tensor并设置损失函数
        # 使用带权重的BCEWithLogitsLoss提高数值稳定性
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1]/class_weights[0]).to(device))
        
        # 修改模型最后一层，使用logits输出
        if hasattr(model, 'sigmoid'):
            model.sigmoid = nn.Identity()  # 移除sigmoid，让BCEWithLogitsLoss内部处理
    else:
        # 其他数据集使用标准BCE损失
        criterion = nn.BCELoss()
    
    # Advanced optimization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduling for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=8
    )
    
    # 确定是否使用logits输出
    use_logits = (dataset_name in ['german', 'xinwang'])
    
    # Train model with advanced techniques
    start_time = time.time()
    trained_model, train_losses, val_accuracies, best_val_acc = trainer.train_model_advanced(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, use_logits
    )
    training_time = time.time() - start_time
    
    # Test model
    test_results = trainer.test_model(trained_model, test_loader, use_logits)
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in trained_model.parameters()) / 1024  # KB
    
    print(
        f"     ✅ {dataset_name.upper()}: Enhanced Neural Network - "
        f"Accuracy: {test_results['accuracy']:.4f}, AUC: {test_results.get('auc', 0.0):.4f}, F1: {test_results['f1']:.4f}"
    )
    
    return {
        'model': trained_model,
        'model_type': 'Enhanced Neural Network',
        'accuracy': test_results['accuracy'],
        'precision': test_results['precision'],
        'recall': test_results['recall'],
        'f1': test_results['f1'],
        'auc': test_results.get('auc', 0.5),
        'predictions': test_results['predictions'],
        'probabilities': test_results['probabilities'],
        'true_labels': test_results['true_labels'],
        'training_time': training_time,
        'model_size': model_size,
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'feature_names': data_dict['feature_names']
    }

def train_all_teacher_models(processed_data):
    """训练所有教师模型"""
    global _device_shown
    
    if not _device_shown:
        print(f"🔧 Using device: {device}")
        _device_shown = True
    
    teacher_models = {}
    datasets = ['uci', 'german', 'australian', 'xinwang']
    
    from tqdm import tqdm
    
    # 使用tqdm显示进度
    for dataset_name in tqdm(datasets, desc="📚 Training Teacher Models", unit="model"):
        if dataset_name in processed_data:
            teacher_models[dataset_name] = create_teacher_model(dataset_name, processed_data)
        else:
            print(f"   ⚠️ {dataset_name.upper()} dataset not found in processed data")
    
    print("✅ Teacher model training completed")
    for dataset_name, model_info in teacher_models.items():
        print(f"  • {dataset_name.upper()}: {model_info['model_type']} - Accuracy: {model_info['accuracy']:.4f}")
    
    return teacher_models

