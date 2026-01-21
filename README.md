# SHAP-KD：面向信用风险评估的可解释知识蒸馏

本仓库配套论文实现：在四个信用风险数据集上进行**基准模型对比**、**自动教师选择**、以及将教师知识蒸馏到**单棵决策树学生**的 **SHAP-KD**（含可选自适应温度变体），并输出可审计的 IF-THEN 规则。

## 主要特性

- **基准模型**：LR/SVM/RF/GBDT/XGBoost/LightGBM/CatBoost（可选 Optuna 调参）
- **神经网络基线**：CreditNet
- **SHAP-KD**：Teacher → Decision Tree Student，使用温度缩放 soft labels + 样本权重训练（`sample_weight`）
- **自动教师选择**：在验证集上基于 AUC 选择最优 Teacher
- **规则导出**：从蒸馏后的树导出路径规则（文本 + Excel）
- **可解释性图**：决策树 + SHAP 重要性图（保存到论文 Figure 目录）

## 数据集

| Dataset | Samples | Features | Source |
|---------|---------|----------|--------|
| German Credit | 1,000 | 20 | UCI |
| Australian Credit | 690 | 14 | UCI |
| Xinwang Credit | 17,884 | 100 | Chinese P2P |
| UCI Credit Card | 30,000 | 23 | UCI |

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 运行实验

**推荐方式：使用解耦的实验流程**

```bash
# 完整流程：训练教师模型（带缓存）+ 蒸馏实验
python run_experiments.py --dataset german

# 仅训练教师模型（结果会被缓存）
python run_experiments.py --dataset german --stage teacher

# 仅运行蒸馏实验（需要已有缓存的教师模型）
python run_experiments.py --dataset german --stage distill

# 强制重新训练教师模型
python run_experiments.py --dataset german --force

# 运行所有数据集
python run_experiments.py --dataset all

# 指定GPU设备
python run_experiments.py --dataset german --device 0

# 快速启动脚本
python quick_start.py german
```

**解耦架构的优势：**
- ✅ 教师模型训练与蒸馏过程分离
- ✅ 训练好的教师模型和SHAP值自动缓存到 `results/teacher_cache/`
- ✅ 重复实验时自动跳过教师训练阶段
- ✅ 支持单独运行教师训练或蒸馏阶段
- ✅ 教师选择结果保存到 Excel 便于查看

**传统方式（仍然支持）：**

```bash
# 单一入口：main.py（建议先用 n-runs=1 做最小验证）
python main.py run --datasets german --n-runs 1

# 四个数据集：german, australian, uci, xinwang
python main.py run --datasets german australian uci xinwang --n-runs 1
```

说明：实验默认输出到 `results/`，并在 `Manuscript_FI/Manuscript_FI/Figure/`（若存在）保存论文用图（SHAP/消融）。

### 3) 单独生成论文 Figure

```bash
# Fig.1：按“各数据集实际使用的 Teacher”重算 Top-10 SHAP 并覆盖保存
python main.py shap --datasets german australian uci xinwang

# Fig.2：生成消融图（若缺少 *_ablation.xlsx，可加 --compute 先计算）
python main.py ablation --datasets german australian uci xinwang --compute --n-runs 1
```

### 4) 规则导出

```bash
python main.py rules --model results/model_cache/<your_dt_model>.pkl --output-dir results/rules
```

## 目录结构（以当前仓库为准）

```
FinancialInnovation/
├── main.py                    # 传统实验入口
├── run_experiments.py         # 解耦实验入口（推荐）
├── quick_start.py             # 快速启动脚本
├── requirements.txt
├── data/
│   ├── german_credit.csv
│   ├── australian_credit.csv
│   ├── xinwang.csv
│   └── uci_credit.csv
├── results/
│   ├── teacher_cache/         # 教师模型和SHAP缓存
│   ├── figures/               # 生成的图表
│   └── rules/                 # 提取的规则
├── Manuscript_FI/
└── src/
    ├── data/preprocessor.py
    ├── models/{baselines.py, neural.py}
    ├── distillation/{dt_distiller.py, losses.py, trainer.py}
    ├── experiments/
    │   ├── runner.py          # 实验运行器
    │   └── teacher_trainer.py # 教师模型训练器
    └── visualization/{plots.py, nature_plots.py}
```

### 基线模型说明

Teacher选择阶段会训练以下所有基线模型：

| 类别 | 模型 | 说明 |
|-----|------|------|
| Linear | LR-Ridge, LR-Lasso, LR-ElasticNet | 逻辑回归 (L2/L1/ElasticNet) |
| Kernel | SVM-RBF, SVM-Linear | 支持向量机 |
| Tree | DT | 决策树 |
| Instance-based | KNN | K近邻 |
| Probabilistic | NB | 朴素贝叶斯 |
| Ensemble (Optuna) | RF-Tuned, GBDT-Tuned, XGBoost-Tuned, LightGBM-Tuned, CatBoost-Tuned | Optuna超参数优化 |
| Neural | CreditNet | 自定义神经网络 |

### 缓存文件说明

教师模型缓存 (`results/teacher_cache/`):
- `{dataset}_teacher_cache.json` - 教师模型元信息（名称、AUC、超参数等）
- `{dataset}_teacher_{model}.pkl` - 训练好的教师模型
- `{dataset}_teacher_shap.npz` - 预计算的SHAP值
- `{dataset}_all_models_index.json` - 所有保存模型的索引

教师选择结果保存到 `results/{dataset}_teacher_selection.xlsx`

## 📐 Theoretical Foundations

### Theorem 1: Temperature-Interpretability Tradeoff

$$\mathbb{E}[\|p_S - p_T\|_2] \leq \frac{C_1}{\sqrt{\tau}} + C_2 \cdot \exp\left(-\frac{\tau}{\tau_0}\right)$$

### Theorem 2: Generalization Bound for SHAP-guided Distillation

$$\epsilon_S \leq \epsilon_T + O\left(\sqrt{\frac{k \cdot \log k}{n}}\right) + O\left(d_{\max}^{-1}\right) + O\left(\frac{1}{\tau}\right)$$

### Theorem 3: Feature Selection Consistency

$$P\left(|S_k \cap S_k^*| \geq (1-\delta)k\right) \geq 1 - 2\exp\left(-\frac{n\delta^2}{2}\right)$$

## 🔬 Baseline Models

| Model | Category | Reference |
|-------|----------|-----------|
| LR-Ridge | Linear | Hosmer & Lemeshow (2000) |
| LR-Lasso | Linear | Tibshirani (1996) |
| LR-ElasticNet | Linear | Zou & Hastie (2005) |
| SVM-RBF | Kernel | Cortes & Vapnik (1995) |
| RF | Ensemble | Breiman (2001) |
| GBDT | Ensemble | Friedman (2001) |
| XGBoost | Ensemble | Chen & Guestrin (2016) |
| LightGBM | Ensemble | Ke et al. (2017) |
| CatBoost | Ensemble | Prokhorenkova et al. (2018) |

## 输出与命名规范（与论文一致）

- 基准表：`results/<dataset>_baseline.xlsx`
- 蒸馏表：`results/<dataset>_distillation.xlsx`，方法名使用论文口径：
  - `Teacher`
  - `Student Baseline (DT)`
  - `VanillaKD`
  - `SoftLabelKD`
  - `SHAP-KD`
  - `SHAP-KD (Adaptive)`（如启用）
- 消融表：`results/<dataset>_ablation.xlsx`
- 规则导出：`results/rules/<dataset>_SHAP-KD_rules.(txt|xlsx)`
- SHAP 重要性图：`Manuscript_FI/Manuscript_FI/Figure/shap_<dataset>_top10.png`（若 Figure 目录存在）

## 常见问题

- Windows 中文路径：已在 SHAP 绘图中默认 `n_jobs=1` 避免并行编码问题。
- CatBoost 生成 `catboost_info/`：已在代码中设置 `allow_writing_files=False`，不再生成该目录。

## 📊 Example Results

### German Credit Dataset - Distillation Results

| Model | AUC | Accuracy | F1 | Interpretable |
|-------|-----|----------|-----|---------------|
| Teacher (Best Baseline) | 0.867 | 0.834 | 0.821 | ❌ |
| DT-Baseline | 0.712 | 0.695 | 0.683 | ✅ |
| **SHAP-KD-DT (Ours)** | **0.845** | **0.812** | **0.798** | ✅ |

### Rule Extraction Example

```
[R1] (Samples: 245, Confidence: 87.35%)
  IF Status_A14 <= 0.23 AND Age <= 2.45 AND Duration <= 0.15
  THEN credit_risk = Non-default

[R2] (Samples: 123, Confidence: 82.11%)
  IF Status_A14 <= 0.23 AND Age <= 2.45 AND Duration > 0.15
  THEN credit_risk = Default
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024sakd,
  title={SHAP-guided Adaptive Knowledge Distillation for Interpretable Credit Scoring},
  author={Author, A. and Author, B.},
  journal={Financial Innovation},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

