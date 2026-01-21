"""
实验追踪与日志系统
Experiment Tracking and Logging System

支持MLflow、WandB、TensorBoard等实验追踪工具
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "results/logs",
        enable_mlflow: bool = True,
        enable_console: bool = True,
        log_level: str = "INFO"
    ):
        """
        初始化实验日志记录器
        
        Args:
            experiment_name: 实验名称
            log_dir: 日志保存目录
            enable_mlflow: 是否启用MLflow
            enable_console: 是否输出到控制台
            log_level: 日志级别
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"
        
        # 设置Python logging
        self.logger = self._setup_logger(log_level, enable_console)
        
        # 设置MLflow
        self.mlflow_enabled = enable_mlflow
        if self.mlflow_enabled:
            self._setup_mlflow()
        
        # 实验元数据
        self.metadata = {
            'experiment_name': experiment_name,
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'parameters': {},
            'metrics': {},
            'artifacts': []
        }
    
    def _setup_logger(self, log_level: str, enable_console: bool) -> logging.Logger:
        """设置Python日志记录器"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 避免重复handler
        logger.handlers.clear()
        
        # 文件handler
        log_file = self.log_dir / f"{self.run_id}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            logger.addHandler(console_handler)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_mlflow(self):
        """设置MLflow追踪"""
        mlflow_dir = self.log_dir.parent / "mlruns"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_id)
        
        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        self.logger.info(f"MLflow experiment: {self.experiment_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """记录超参数"""
        self.metadata['parameters'].update(params)
        
        if self.mlflow_enabled:
            mlflow.log_params(params)
        
        self.logger.info(f"Logged parameters: {params}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None
    ):
        """记录评估指标"""
        for key, value in metrics.items():
            if key not in self.metadata['metrics']:
                self.metadata['metrics'][key] = []
            self.metadata['metrics'][key].append({
                'value': value,
                'step': step,
                'timestamp': datetime.now().isoformat()
            })
        
        if self.mlflow_enabled:
            mlflow.log_metrics(metrics, step=step)
        
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in metrics.items()])
        self.logger.info(f"Metrics (step={step}): {metrics_str}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """记录实验产物（模型、图表等）"""
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            self.logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        if artifact_name is None:
            artifact_name = artifact_path.name
        
        self.metadata['artifacts'].append({
            'name': artifact_name,
            'path': str(artifact_path),
            'timestamp': datetime.now().isoformat()
        })
        
        if self.mlflow_enabled:
            mlflow.log_artifact(str(artifact_path))
        
        self.logger.info(f"Logged artifact: {artifact_name}")
    
    def log_model(self, model, model_name: str, metadata: Optional[Dict] = None):
        """记录模型"""
        if self.mlflow_enabled:
            # 保存sklearn模型
            if hasattr(model, 'predict'):
                mlflow.sklearn.log_model(model, model_name)
            # 保存PyTorch模型
            elif hasattr(model, 'state_dict'):
                mlflow.pytorch.log_model(model, model_name)
        
        self.logger.info(f"Logged model: {model_name}")
        
        if metadata:
            self.log_params({f"{model_name}_{k}": v for k, v in metadata.items()})
    
    def log_figure(self, figure, figure_name: str):
        """记录matplotlib图表"""
        figure_path = self.log_dir / f"{figure_name}.png"
        figure.savefig(figure_path, dpi=300, bbox_inches='tight')
        
        if self.mlflow_enabled:
            mlflow.log_figure(figure, figure_name)
        
        self.logger.info(f"Logged figure: {figure_name}")
    
    def log_table(self, df: pd.DataFrame, table_name: str):
        """记录数据表格"""
        table_path = self.log_dir / f"{table_name}.csv"
        df.to_csv(table_path, index=False)
        
        self.log_artifact(str(table_path), table_name)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def save_metadata(self):
        """保存实验元数据"""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        metadata_path = self.log_dir / f"{self.run_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
    
    def finish(self):
        """结束实验追踪"""
        self.save_metadata()
        
        if self.mlflow_enabled:
            mlflow.end_run()
        
        self.logger.info(f"Experiment {self.run_id} finished")


class MetricsTracker:
    """指标追踪器 - 用于多次运行的统计分析"""
    
    def __init__(self):
        self.runs = []
        self.current_run = {}
    
    def start_run(self, run_id: str):
        """开始新的运行"""
        self.current_run = {
            'run_id': run_id,
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def add_metric(self, name: str, value: float):
        """添加指标"""
        self.current_run['metrics'][name] = value
    
    def end_run(self):
        """结束当前运行"""
        self.runs.append(self.current_run.copy())
        self.current_run = {}
    
    def get_statistics(self) -> pd.DataFrame:
        """计算统计量（均值、标准差、置信区间）"""
        if not self.runs:
            return pd.DataFrame()
        
        # 提取所有指标
        all_metrics = {}
        for run in self.runs:
            for metric_name, value in run['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # 计算统计量
        stats = []
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            stats.append({
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'ci_95_lower': np.percentile(values, 2.5),
                'ci_95_upper': np.percentile(values, 97.5),
                'num_runs': len(values)
            })
        
        return pd.DataFrame(stats)
    
    def save_results(self, output_path: str):
        """保存结果"""
        # 保存所有运行
        runs_df = pd.DataFrame(self.runs)
        runs_path = Path(output_path).parent / f"{Path(output_path).stem}_all_runs.csv"
        runs_df.to_csv(runs_path, index=False)
        
        # 保存统计摘要
        stats_df = self.get_statistics()
        stats_df.to_csv(output_path, index=False)
        
        return stats_df
