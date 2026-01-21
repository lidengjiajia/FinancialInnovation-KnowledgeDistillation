#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP-KD: 基于SHAP的知识蒸馏信用风险评估框架
============================================

解耦架构：
  Stage 1: 教师模型准备 (Optuna调优 + SHAP计算 + 缓存)
  Stage 2: 知识蒸馏实验 (使用缓存的教师模型)

使用方法:
    python main.py                             # 运行全部数据集
    python main.py --dataset german            # 单个数据集
    python main.py --dataset german --stage teacher   # 仅训练教师
    python main.py --dataset german --stage distill   # 仅蒸馏实验
    python main.py --dataset german --force    # 强制重新训练教师
    python main.py --n-runs 1                  # 快速测试
"""

import os
import sys
import argparse
import logging
import warnings
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import torch

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 确保能导入src包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optuna日志抑制
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# 配置
# ============================================================================
SEED = 42
DATASETS = ['german', 'australian', 'uci', 'xinwang']
RESULTS_DIR = 'results'
DATA_DIR = 'data'
FIGURE_DIR = os.path.join(RESULTS_DIR, 'figures')


def setup_logging():
    """配置日志"""
    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    log_file = os.path.join(RESULTS_DIR, 'logs', f"exp_{datetime.now():%Y%m%d_%H%M%S}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger('SHAP-KD')


def print_banner():
    """打印程序信息"""
    print('\n' + '=' * 70)
    print('  SHAP-KD: 基于SHAP的知识蒸馏信用风险评估框架')
    print('  Decoupled Architecture: Teacher Preparation → Distillation')
    print('=' * 70)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'  GPU: {gpu} ({mem:.1f} GB)')
    else:
        print(f'  GPU: 不可用，使用CPU')
    
    print(f'  PyTorch: {torch.__version__}')
    print(f'  时间: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('=' * 70 + '\n')


class ExperimentPipeline:
    """
    解耦的实验流水线
    
    Stage 1: 教师模型准备
        - Optuna调优集成模型 (RF, XGBoost, LightGBM, CatBoost, GBDT)
        - 训练CreditNet神经网络 (100 epochs)
        - 选择最优教师 (基于验证集AUC)
        - 计算SHAP值并生成图表
        - 缓存所有结果
    
    Stage 2: 知识蒸馏实验
        - 加载缓存的教师模型和SHAP值
        - 运行蒸馏实验 (VanillaKD, SoftLabelKD, SHAP-KD)
        - 运行消融实验
    """
    
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        results_dir: str = RESULTS_DIR,
        seed: int = SEED,
        n_runs: int = 5,
        n_optuna_trials: int = 50,
        nn_epochs: int = 100,
        device_id: int = None,
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed = seed
        self.n_runs = n_runs
        self.n_optuna_trials = n_optuna_trials
        self.nn_epochs = nn_epochs
        
        # GPU设置
        self.use_gpu = torch.cuda.is_available()
        if device_id is not None and self.use_gpu:
            torch.cuda.set_device(device_id)
            self.device_id = device_id
        else:
            self.device_id = 0 if self.use_gpu else None
        
        self.logger = logging.getLogger('SHAP-KD')
        
        # 创建目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        
        # 延迟加载组件
        self._teacher_trainer = None
        self._experiment_runner = None
    
    @property
    def teacher_trainer(self):
        """延迟加载TeacherTrainer"""
        if self._teacher_trainer is None:
            from src.experiments.teacher_trainer import TeacherTrainer
            self._teacher_trainer = TeacherTrainer(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
                seed=self.seed,
                n_optuna_trials=self.n_optuna_trials,
                nn_epochs=self.nn_epochs,
                use_gpu=self.use_gpu,
                device_id=self.device_id
            )
        return self._teacher_trainer
    
    @property
    def experiment_runner(self):
        """延迟加载ExperimentRunner"""
        if self._experiment_runner is None:
            from src.experiments.runner import ExperimentRunner
            self._experiment_runner = ExperimentRunner(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
                n_runs=self.n_runs,
                seed=self.seed,
                device_id=self.device_id,
                force_retrain=False,
                generate_shap_plots=False,  # SHAP图在Stage 1生成
                use_gpu_for_baseline=self.use_gpu
            )
        return self._experiment_runner
    
    def check_teacher_cache(self, dataset_name: str) -> bool:
        """检查教师缓存是否存在"""
        cache = self.teacher_trainer.check_teacher_cache(dataset_name)
        return cache is not None
    
    def run_stage1_teacher_preparation(
        self,
        dataset_name: str,
        force_retrain: bool = False
    ) -> Dict:
        """
        Stage 1: 教师模型准备
        
        包括：Optuna调优、CreditNet训练、SHAP计算、生成图表
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STAGE 1: 教师模型准备 - {dataset_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        # 检查缓存
        if not force_retrain:
            cache = self.teacher_trainer.check_teacher_cache(dataset_name)
            if cache is not None:
                self.logger.info(f"[OK] 使用缓存的教师: {cache.teacher_info.name}")
                self.logger.info(f"  验证集AUC: {cache.teacher_info.auc_val:.4f}")
                self.logger.info(f"  测试集AUC: {cache.teacher_info.auc_test:.4f}")
                if cache.shap_plot_path and os.path.exists(cache.shap_plot_path):
                    self.logger.info(f"  SHAP图: {cache.shap_plot_path}")
                return {
                    'cached': True,
                    'teacher_info': cache.teacher_info,
                    'cache': cache
                }
        
        # 训练并选择教师
        start_time = time.time()
        cache = self.teacher_trainer.train_and_select_teacher(
            dataset_name,
            force_retrain=force_retrain,
            compute_shap=True
        )
        elapsed = time.time() - start_time
        
        self.logger.info(f"\n✓ 教师准备完成 ({elapsed/60:.1f} 分钟)")
        self.logger.info(f"  最优教师: {cache.teacher_info.name}")
        self.logger.info(f"  验证集AUC: {cache.teacher_info.auc_val:.4f}")
        self.logger.info(f"  测试集AUC: {cache.teacher_info.auc_test:.4f}")
        self.logger.info(f"  模型保存到: {cache.model_path}")
        if cache.shap_values_path:
            self.logger.info(f"  SHAP值保存到: {cache.shap_values_path}")
        if cache.shap_plot_path:
            self.logger.info(f"  SHAP图保存到: {cache.shap_plot_path}")
        
        return {
            'cached': False,
            'teacher_info': cache.teacher_info,
            'cache': cache,
            'elapsed': elapsed
        }
    
    def run_stage2_distillation(self, dataset_name: str) -> Dict:
        """
        Stage 2: 知识蒸馏实验
        
        使用缓存的教师模型运行蒸馏和消融实验
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STAGE 2: 知识蒸馏实验 - {dataset_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        # 检查教师缓存
        cache = self.teacher_trainer.check_teacher_cache(dataset_name)
        if cache is None:
            self.logger.warning(f"未找到教师缓存，先运行Stage 1...")
            self.run_stage1_teacher_preparation(dataset_name)
            cache = self.teacher_trainer.check_teacher_cache(dataset_name)
        
        self.logger.info(f"使用教师: {cache.teacher_info.name} (AUC={cache.teacher_info.auc_val:.4f})")
        
        # 加载教师模型
        teacher_model, teacher_info, is_neural = self.teacher_trainer.load_teacher_model(dataset_name)
        
        # 加载SHAP值
        shap_data = self.teacher_trainer.load_teacher_shap(dataset_name)
        precomputed_shap = None
        if shap_data is not None:
            precomputed_shap = shap_data['shap_values']
            self.logger.info(f"已加载SHAP值: shape={precomputed_shap.shape}")
            # 缓存到runner
            self.experiment_runner._cached_shap_values = precomputed_shap
        
        # 注入教师信息到runner
        self._inject_teacher_info(dataset_name, cache, teacher_model, is_neural)
        
        start_time = time.time()
        
        # 运行蒸馏实验 (precomputed_shap已通过_cached_shap_values注入)
        self.logger.info("\n运行蒸馏实验...")
        dist_results = self.experiment_runner.run_distillation_experiments(dataset_name)
        
        # 运行消融实验
        self.logger.info("\n运行消融实验...")
        ablation_results = self.experiment_runner.run_ablation_experiments(dataset_name)
        
        elapsed = time.time() - start_time
        self.logger.info(f"\n✓ 蒸馏实验完成 ({elapsed/60:.1f} 分钟)")
        
        return {
            'distillation': dist_results,
            'ablation': ablation_results,
            'elapsed': elapsed
        }
    
    def _inject_teacher_info(self, dataset_name, cache, teacher_model, is_neural):
        """将教师信息注入到ExperimentRunner"""
        if not hasattr(self.experiment_runner, '_optimization_results'):
            self.experiment_runner._optimization_results = {}
        
        if not is_neural:
            self.experiment_runner._optimization_results[cache.teacher_info.name] = {
                'best_score': cache.teacher_info.auc_val,
                'best_model': teacher_model,
                'best_params': cache.teacher_info.hyperparams
            }
        
        # 设置缓存的教师信息 (包含所有必要字段)
        self.experiment_runner._cached_teacher = {
            'model': teacher_model,
            'name': cache.teacher_info.name,
            'auc': cache.teacher_info.auc_val,
            'is_neural': is_neural,
            'info': cache.teacher_info,
            'dataset': dataset_name
        }
    
    def run_full_pipeline(
        self,
        dataset_name: str,
        force_retrain: bool = False
    ) -> Dict:
        """运行完整流水线"""
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"# 完整流水线: {dataset_name.upper()}")
        self.logger.info(f"{'#'*70}")
        
        results = {'dataset': dataset_name}
        
        # Stage 1
        stage1_result = self.run_stage1_teacher_preparation(dataset_name, force_retrain)
        results['teacher'] = {
            'name': stage1_result['teacher_info'].name,
            'type': stage1_result['teacher_info'].model_type,
            'auc_val': stage1_result['teacher_info'].auc_val,
            'auc_test': stage1_result['teacher_info'].auc_test,
        }
        
        # Stage 2
        stage2_result = self.run_stage2_distillation(dataset_name)
        results['distillation'] = stage2_result['distillation']
        results['ablation'] = stage2_result['ablation']
        
        self.logger.info(f"\n✓ {dataset_name.upper()} 完整流水线完成")
        
        return results
    
    def run_all_datasets(
        self,
        datasets: List[str],
        stage: str = 'all',
        force_retrain: bool = False
    ) -> Dict[str, Dict]:
        """运行多个数据集
        
        Args:
            datasets: 数据集列表
            stage: 运行阶段 ('teacher', 'distill', 'all')
            force_retrain: 是否强制重新训练
        """
        all_results = {}
        
        for i, dataset in enumerate(datasets, 1):
            self.logger.info(f"\n[{i}/{len(datasets)}] 处理数据集: {dataset.upper()}")
            
            try:
                if stage == 'teacher':
                    result = self.run_stage1_teacher_preparation(dataset, force_retrain)
                    all_results[dataset] = {'teacher': result}
                elif stage == 'distill':
                    result = self.run_stage2_distillation(dataset)
                    all_results[dataset] = result
                else:
                    all_results[dataset] = self.run_full_pipeline(dataset, force_retrain)
            except Exception as e:
                self.logger.error(f"处理 {dataset} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成汇总报告
        self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, all_results: Dict):
        """生成汇总报告"""
        import pandas as pd
        
        rows = []
        for dataset, result in all_results.items():
            if 'teacher' in result:
                teacher = result['teacher']
                if isinstance(teacher, dict) and 'name' in teacher:
                    rows.append({
                        'dataset': dataset,
                        'teacher': teacher['name'],
                        'type': teacher.get('type', ''),
                        'val_auc': teacher.get('auc_val', ''),
                        'test_auc': teacher.get('auc_test', '')
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            summary_path = os.path.join(self.results_dir, 'experiment_summary.xlsx')
            df.to_excel(summary_path, index=False)
            self.logger.info(f"\n汇总报告已保存到: {summary_path}")


def generate_ablation_figures(datasets: List[str], logger):
    """生成消融实验图表"""
    try:
        from src.visualization.ablation_figures import generate_ablation_figures as gen_figs
        from src.visualization.ablation_figures import generate_rule_effectiveness_figure
        from src.visualization.ablation_figures import generate_class_balance_figure
        
        saved = gen_figs(
            results_dir=RESULTS_DIR,
            output_dir=FIGURE_DIR,
            datasets=datasets
        )
        if saved:
            logger.info(f"  ✓ 生成 {len(saved)} 张消融图")
        else:
            logger.warning(f"  未生成消融图 (缺少 *_ablation.xlsx)")
        
        # Generate rule effectiveness figure
        rule_fig = generate_rule_effectiveness_figure(
            rules_dir=os.path.join(RESULTS_DIR, 'rules'),
            output_dir=FIGURE_DIR,
            datasets=datasets
        )
        if rule_fig:
            logger.info(f"  ✓ 生成规则有效性图: {rule_fig}")
        
        # Generate class balance ablation figure
        cb_fig = generate_class_balance_figure(
            results_dir=RESULTS_DIR,
            output_dir=FIGURE_DIR,
            datasets=datasets
        )
        if cb_fig:
            logger.info(f"  ✓ 生成类别平衡消融图: {cb_fig}")
    except Exception as e:
        logger.warning(f"  图表生成失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='SHAP-KD: 基于SHAP的知识蒸馏信用风险评估框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 运行全部数据集
  python main.py --dataset german                   # 单个数据集
  python main.py --dataset german --stage teacher   # 仅训练教师
  python main.py --dataset german --stage distill   # 仅蒸馏实验
  python main.py --dataset german --force           # 强制重新训练
  python main.py --n-runs 1                         # 快速测试
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='all',
        choices=['german', 'australian', 'uci', 'xinwang', 'all'],
        help='数据集 (默认: all)'
    )
    
    parser.add_argument(
        '--stage', '-s',
        type=str,
        default='all',
        choices=['teacher', 'distill', 'all'],
        help='运行阶段: teacher(教师准备), distill(蒸馏), all(全部)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新训练教师模型'
    )
    
    parser.add_argument(
        '--n-runs', '-n',
        type=int,
        default=5,
        help='实验重复次数 (默认: 5)'
    )
    
    parser.add_argument(
        '--n-trials', '-t',
        type=int,
        default=50,
        help='Optuna调优次数 (默认: 50)'
    )
    
    parser.add_argument(
        '--nn-epochs',
        type=int,
        default=100,
        help='神经网络训练轮数 (默认: 100)'
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='GPU设备ID'
    )
    
    parser.add_argument(
        '--figures-only',
        action='store_true',
        help='仅生成图表'
    )
    
    args = parser.parse_args()
    
    # 初始化
    print_banner()
    logger = setup_logging()
    
    # 设置随机种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # 确定数据集
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    
    logger.info(f'数据集: {datasets}')
    logger.info(f'运行阶段: {args.stage}')
    logger.info(f'重复次数: {args.n_runs}')
    logger.info(f'Optuna试验: {args.n_trials}')
    logger.info(f'强制重训: {args.force}')
    
    start_time = datetime.now()
    
    if args.figures_only:
        # 仅生成图表
        logger.info('\n生成消融实验图表...')
        generate_ablation_figures(datasets, logger)
    else:
        # 运行实验
        pipeline = ExperimentPipeline(
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            seed=SEED,
            n_runs=args.n_runs,
            n_optuna_trials=args.n_trials,
            nn_epochs=args.nn_epochs,
            device_id=args.device
        )
        
        results = pipeline.run_all_datasets(
            datasets=datasets,
            stage=args.stage,
            force_retrain=args.force
        )
        
        # 生成消融图
        if args.stage in ['all', 'distill']:
            logger.info('\n生成消融实验图表...')
            generate_ablation_figures(datasets, logger)
    
    elapsed = datetime.now() - start_time
    logger.info(f'\n{"="*60}')
    logger.info(f'总耗时: {elapsed}')
    logger.info(f'结果目录: {RESULTS_DIR}')
    logger.info(f'图表目录: {FIGURE_DIR}')
    logger.info(f'{"="*60}')
    logger.info('完成!')


if __name__ == '__main__':
    main()
