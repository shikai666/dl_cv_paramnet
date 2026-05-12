 # DL-CV ParamNet

## 基于自监督对比学习的深度学习控制CV参数化架构

**DL-CV ParamNet: A Contrastive Self-Supervised Framework for Deep Learning-Controlled CV Parameterization in Liye Qin Bamboo Slips Text Segmentation**

---

## 项目概述

针对里耶秦简灰度图像（从PDF扫描文档提取，无纹理干扰）的文字轮廓检测任务，DL-CV ParamNet 是一个无监督混合架构。该架构通过深度学习（DL）模型自适应优化计算机视觉（CV）操作的参数，实现端到端学习，而无需像素级标注。

### 核心创新

1. **DL 控制 CV 参数**：深度学习网络预测 CV 操作的最优参数，而非直接预测分割结果
2. **自监督学习**：对比学习 + 重建损失，无需像素级标注
3. **端到端可微**：所有 CV 操作使用可微近似，支持梯度回传

---

## 数学框架

### 整体架构

输入灰度图像 $I \in [0, 255]^{H \times W \times 1}$，架构包括四个模块：

```
I → [特征提取器 f_θ] → F' → [参数预测器 g_φ] → P → [CV执行模块 h] → C
                                    ↑                         ↓
                            [自监督训练模块] ←←←←←←←←←←←←←←←←←←←
```

### 1. 特征提取器

$$f_{\theta}: \mathbb{R}^{H \times W \times 1} \to \mathbb{R}^{D}$$

- 预处理：直方图均衡化 $\text{HistEq}(I)_{i,j} = 255 \cdot \sum_{k=0}^{I_{i,j}} p(k)$
- 骨干网络：ResNet-18（修改首层为单通道输入）
- 通道注意力：$F' = F \odot \sigma( W_a \cdot \text{GAP}(F) + b_a )$

### 2. 参数预测器

$$g_{\phi}: \mathbb{R}^{D} \to \mathbb{R}^{K}$$

MLP 预测并映射到物理范围：

- 低阈值：$t_l = \alpha_l \cdot \sigma(p_1) + \beta_l$，其中 $\alpha_l=100, \beta_l=0 \Rightarrow t_l \in [0,100]$
- 高阈值：$t_h = \alpha_h \cdot \sigma(p_2) + \beta_h + t_l$，其中 $\alpha_h=100, \beta_h=50 \Rightarrow t_h > t_l$
- 内核大小：$k_m = \lfloor \alpha_k \cdot \sigma(p_3) + \beta_k \rceil$，其中 $\alpha_k=10, \beta_k=3 \Rightarrow k_m \in [3,13]$

### 3. CV 执行模块

$$h(P, I) = \text{FindContours}( \text{Closing}( \text{Canny}(I, t_l, t_h), k_m ) )$$

- Canny 边缘检测：$G = \sqrt{ (\nabla_x I)^2 + (\nabla_y I)^2 }$
- 闭运算：$\text{Closing}(E, k_m) = \text{Erode}( \text{Dilate}(E, K_{k_m}), K_{k_m} )$

### 4. 自监督训练

**对比损失 (修改版 NT-Xent)**:

$$\mathcal{L}_{cont} = \frac{1}{B} \sum_{b=1}^{B} -\log \frac{ \exp( \text{sim}(z_b^{+}, z_b^{++}) / \tau ) }{ \sum_{k=1, k \neq b}^{B} \left[ \exp( \text{sim}(z_b^{+}, z_k^{-}) / \tau ) + \exp( \text{sim}(z_b^{++}, z_k^{+}) / \tau ) \right] }$$

其中：
- $z = \text{Proj}(\text{Flatten}(C))$，投影函数为两层 MLP
- $\text{Proj}(x) = W_p^{(2)} \rho( W_p^{(1)} x + b_p^{(1)} ) + b_p^{(2)}$
- $\rho$ 为 ReLU 激活函数，$x = \text{Flatten}(C) \in \mathbb{R}^{H \times W}$
- $\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$ 为余弦相似度
- $\tau = 0.1$ 为温度参数
- $z^+, z^{++}$ 为同一图像的两个增强视图

**重建损失**:
$$\mathcal{L}_{rec} = \| C - S(I) \|_1 + \gamma \| \nabla C - \nabla I \|_2^2$$

**总损失**:
$$\mathcal{L} = \lambda_1 \mathcal{L}_{cont} + \lambda_2 \mathcal{L}_{rec}$$

默认超参数：$\tau=0.1, \gamma=0.5, \lambda_1=1.0, \lambda_2=0.8$

---

## 项目结构

```
dl_cv_paramnet/
├── README.md                     # 项目文档
├── requirements.txt              # 依赖列表
├── run.py                        # 主入口脚本
│
├── configs/                      # 配置文件目录
│   └── default.json              # 默认配置
│
├── data/                         # 数据集目录
│   ├── README.md                 # 数据集说明
│   ├── raw/                      # 原始数据（PDF、未处理图像）
│   └── processed/                # 处理后的数据集
│       ├── train/images/         # 训练集
│       ├── val/images/           # 验证集
│       └── test/images/          # 测试集
│
├── src/                          # 源代码
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── losses.py                 # 损失函数
│   ├── data.py                   # 数据加载
│   ├── metrics.py                # 评估指标
│   └── models/                   # 模型模块
│       ├── __init__.py
│       ├── feature_extractor.py  # 特征提取器
│       ├── parameter_predictor.py # 参数预测器
│       ├── cv_execution.py       # CV执行模块
│       └── model.py              # 完整模型
│
├── scripts/                      # 脚本目录
│   ├── train.py                  # 训练脚本
│   ├── evaluate.py               # 评估脚本
│   ├── inference.py              # 推理脚本
│   ├── visualization.py          # 可视化工具
│   ├── ablation.py               # 消融实验
│   ├── prepare_data.py           # 数据准备
│   └── test_all.py               # 测试脚本
│
├── checkpoints/                  # 模型检查点
├── logs/                         # 训练日志
├── outputs/                      # 输出结果
└── experiments/                  # 实验结果
    ├── ablation/                 # 消融实验
    └── sensitivity/              # 敏感性分析
```

---

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (推荐)

### 安装依赖

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm scipy
pip install pdf2image  # 可选，用于从 PDF 提取图像
```

---

## 使用方法

### 1. 数据准备

```bash
# 从 PDF 提取图像
python prepare_data.py extract --pdf document.pdf --output ./extracted

# 预处理和划分数据集
python prepare_data.py prepare --input ./images --output ./dataset

# 创建合成数据集（测试用）
python prepare_data.py synthetic --output ./synthetic_data --num_samples 500

# 分析数据集统计信息
python prepare_data.py analyze --data_dir ./dataset
```

### 2. 训练

```bash
# 使用合成数据快速测试
python train.py --synthetic --epochs 10 --batch_size 16

# 使用真实数据训练
python train.py --data_dir ./dataset --epochs 100 --device cuda

# 恢复训练
python train.py --resume checkpoints/dl_cv_paramnet/best.pth

# 自定义配置
python train.py --config config.json --epochs 200 --lr 5e-5
```

### 3. 推理

```bash
# 单图像推理
python inference.py --checkpoint model.pth --input image.png --output result.png

# 批量处理
python inference.py --checkpoint model.pth --input_dir images/ --output_dir results/

# 生成叠加可视化
python inference.py --checkpoint model.pth --input image.png --overlay

# 导出轮廓到 JSON
python inference.py --checkpoint model.pth --input image.png --export_json
```

### 4. 评估

```bash
# 使用合成数据评估
python evaluate.py --checkpoint model.pth --synthetic

# 使用测试集评估
python evaluate.py --checkpoint model.pth --data_dir ./dataset --has_gt

# 与基线方法比较
python evaluate.py --checkpoint model.pth --data_dir ./dataset
```

### 5. 消融实验

```bash
# 运行所有组件消融
python ablation.py --output_dir ./ablation_results --epochs 50

# 使用合成数据快速测试
python ablation.py --synthetic --epochs 10

# 仅超参数敏感性分析
python ablation.py --mode sensitivity --epochs 30
```

### 6. 可视化

```bash
# 单图像可视化
python visualization.py --mode single --input image.png --contour contour.png --output viz.png

# 参数分布可视化
python visualization.py --mode params --params_file predicted_params.json --output ./visualizations
```

---

## 评价指标

### 定量指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **BF-Score** | $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ | 边界匹配精度，容差 δ=2 像素 |
| **IoU** | $\frac{\|C \cap G\|}{\|C \cup G\|}$ | 区域重叠程度 |
| **Dice** | $\frac{2 \cdot \|C \cap G\|}{\|C\| + \|G\|}$ | 对小目标敏感 |
| **重建误差** | $\|C - S(I)\|_1 + \gamma \|\nabla C - \nabla I\|_2^2$ | 无监督代理指标 |

### 消融实验设计

| 配置 | 描述 |
|------|------|
| Full Model | 完整模型（基准） |
| w/o $\mathcal{L}_{cont}$ | 移除对比损失 |
| w/o $\mathcal{L}_{rec}$ | 移除重建损失 |
| Fixed CV Params | 固定 CV 参数（不使用 DL 预测） |
| w/o Attention | 禁用注意力机制 |

---

## 配置说明

### 模型配置 (ModelConfig)

```python
@dataclass
class ModelConfig:
    feature_dim: int = 512      # 特征维度 D
    predictor_hidden_dim: int = 256  # MLP 隐藏层维度
    
    # 参数映射超参数
    alpha_l: float = 100.0      # t_l 缩放
    beta_l: float = 0.0         # t_l 偏移
    alpha_h: float = 100.0      # t_h 缩放
    beta_h: float = 50.0        # t_h 偏移（保证 t_h > t_l）
    alpha_k: float = 10.0       # k_m 缩放
    beta_k: float = 3.0         # k_m 偏移
```

### 训练配置 (TrainingConfig)

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # 损失权重
    lambda_contrastive: float = 1.0   # λ_1
    lambda_reconstruction: float = 0.8 # λ_2
    temperature: float = 0.1           # τ
    gamma: float = 0.5                 # γ (梯度损失权重)
```

---

## 技术细节

### 可微 CV 操作

为实现端到端训练，所有 CV 操作使用可微近似：

1. **软阈值**：用 sigmoid 替代硬阈值
   $$\text{soft\_threshold}(x, t) = \sigma(k \cdot (x - t))$$

2. **软 NMS**：用 softmax 近似非极大抑制
   
3. **形态学操作**：
   - 软膨胀：$\text{soft\_dilate}(x) = \text{softmax}(x \cdot T) \cdot x$
   - 软腐蚀：$\text{soft\_erode}(x) = \text{softmax}(-x \cdot T) \cdot x$

4. **STE (Straight-Through Estimator)**：前向用离散值，反向传播用连续梯度

### 数据增强策略

仅使用几何变换（匹配干净灰度图特性，避免噪声增强）：

- 旋转：$[-15°, +15°]$
- 缩放：$[0.8, 1.2]$
- 水平/垂直翻转

---

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@article{dlcvparamnet2024,
  title={DL-CV ParamNet: A Contrastive Self-Supervised Framework for Deep Learning-Controlled CV Parameterization in Liye Qin Bamboo Slips Text Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请提交 Issue 或联系作者。
