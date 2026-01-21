"""
配置管理系统
Configuration Management System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    name: str
    version: str
    random_seed: int
    device: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """配置管理器 - 统一管理所有实验配置"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 config/experiment_config.yaml
        """
        if config_path is None:
            # 自动查找配置文件
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "experiment_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请确保 config/experiment_config.yaml 存在"
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 环境变量替换
        config = self._resolve_env_variables(config)
        
        # 路径解析
        config = self._resolve_paths(config)
        
        return config
    
    def _resolve_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解析环境变量"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("$"):
                env_var = value[1:]
                return os.environ.get(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
        
        return resolve_value(config)
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解析相对路径为绝对路径"""
        project_root = Path(__file__).parent.parent.parent
        
        def resolve_path(value):
            if isinstance(value, str) and ('/' in value or '\\' in value):
                path = Path(value)
                if not path.is_absolute():
                    return str(project_root / path)
            elif isinstance(value, dict):
                return {k: resolve_path(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_path(item) for item in value]
            return value
        
        return resolve_path(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号分隔的嵌套访问）
        
        Example:
            config.get('baseline_models.traditional[0].name')
            config.get('distillation.strategies')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            elif isinstance(value, list):
                try:
                    # 支持列表索引: key[0]
                    if '[' in k and ']' in k:
                        list_key = k[:k.index('[')]
                        index = int(k[k.index('[')+1:k.index(']')])
                        value = value[index]
                    else:
                        value = default
                except (ValueError, IndexError):
                    value = default
            else:
                return default
                
            if value is None:
                return default
        
        return value
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """获取特定数据集的配置"""
        return self.config.get('datasets', {}).get(dataset_name, {})
    
    def get_baseline_models(self) -> list:
        """获取基线模型配置列表"""
        traditional = self.config.get('baseline_models', {}).get('traditional', [])
        deep_learning = self.config.get('baseline_models', {}).get('deep_learning', [])
        return traditional + deep_learning
    
    def get_distillation_strategies(self) -> list:
        """获取蒸馏策略配置"""
        return self.config.get('distillation', {}).get('strategies', [])
    
    def get_student_models(self) -> list:
        """获取学生模型配置"""
        return self.config.get('distillation', {}).get('student_models', [])
    
    def save_snapshot(self, output_path: str):
        """保存配置快照（用于实验可复现性）"""
        snapshot = {
            'config': self.config,
            'config_path': str(self.config_path),
            'timestamp': self._get_timestamp()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def validate(self) -> bool:
        """验证配置文件的完整性"""
        required_sections = [
            'global',
            'datasets',
            'baseline_models',
            'distillation',
            'evaluation_metrics'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需部分: {section}")
        
        return True
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


# 单例模式
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器实例（单例）"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    
    return _config_instance
