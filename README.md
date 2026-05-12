# DL-CV ParamNet

## Self-Supervised Parameterization of Classical CV for Jiandu Manuscript Mask Extraction

**DL-CV ParamNet: An Interpretable Hybrid Framework for Foreground-Mask Extraction on Jiandu Bamboo-Slip Manuscripts via Self-Supervised Learning of Classical-CV Control Parameters**

> 论文配套代码与数据。Paper: see `paper/DL_CV_ParamNet_revised.pdf` (or the arXiv link, once available).

---

## 项目概述 / Overview

DL-CV ParamNet 是一个**可解释的混合框架**，用于在标注稀缺的历史简牍 (Jiandu) 文档上提取前景文字 mask。与传统的端到端深度分割网络不同，本方法**只让深度学习预测三个可解释的经典 CV 参数**——Canny 双阈值 $(t_l, t_h)$ 和形态学闭运算的核大小 $k_m$——而轮廓生成 (Canny + Closing) 与最终 mask 构造 (overlap-based foreground selection) 全程保持经典、确定、可诊断。

主战场是公开的 **DeepJiandu** 数据集（5,922 张训练图、100 张手工 mask 标注的评估子集）。模型在此完全自监督训练，不使用任何像素级、字符级或边界级标注。此外提供一个**里耶秦简零样本泛化分析**作为跨简牍源的迁移验证。

### 核心定位 / Positioning

1. **DL 控制 CV 参数，不预测像素**：网络只学一个三维控制向量 $(t_l, t_h, k_m)$，避免在无像素标注下学习高维 dense mask；
2. **完全无监督训练**：NT-Xent 对比一致性 + Sobel 伪边重建，两项自监督目标合作，无需任何标签；
3. **评估期确定性**：从 contour cue 到 final mask 全部经典几何规则，每一次失败都能定位到具体 stage (thresholding / closing / overlap filtering / mask-to-box)。

---

## 数学框架 / Mathematical Framework

输入灰度图像 $x \in \mathbb{R}^{H \times W}$，先用 CLAHE 进行对比度增强 $x' = \mathrm{CLAHE}(x)$，再进入下列流水线。

### 1. 参数预测网络 (Parameter Prediction Network)

对增强图 $x'$ 的两个增强视图 $x_1, x_2$，共享权重的 ResNet-18 + SE channel attention 提取全局特征：

$$F_i' = F_i \odot \sigma(W_a \cdot \mathrm{GAP}(F_i) + b_a), \quad F_i = f_\Theta(x_i)$$

MLP 头 $g_\Phi$ 输出三维 raw $p_i = [p_{1,i}, p_{2,i}, p_{3,i}]$，经有界映射约束到合法 CV 参数范围：

$$t_{l,i} = 100\,\sigma(p_{1,i}), \quad t_{l,i} \in [0, 100]$$

$$t_{h,i} = t_{l,i} + (200 - t_{l,i})\,\sigma(p_{2,i}), \quad t_{h,i} \in [t_{l,i}, 200]$$

$$k_{m,i} = 10\,\sigma(p_{3,i}) + 3, \quad k_{m,i} \in [3, 13]$$

评估期 $k_m$ 离散化到最近的奇数 $\{3, 5, 7, 9, 11, 13\}$。

### 2. 经典 CV 执行模块 (Classical CV Execution)

$$E_i = \mathrm{Canny}(x_i; t_{l,i}, t_{h,i})$$

$$M_{c,i} = \mathrm{Closing}(E_i; k_{m,i}) = \mathrm{Erode}(\mathrm{Dilate}(E_i, K_{k_{m,i}}), K_{k_{m,i}})$$

得到中间二值 contour cue $M_{c,i} \in \{0, 1\}^{H \times W}$。

### 3. 确定性 Overlap-Mask 生成 (Deterministic Overlap-Mask Generation, $\mathcal{G}$)

**仅在评估期对单视图使用**，将 contour cue $M_c$ 转为最终 mask $M_o$：

- **Loose ink candidates**: 对原图 $x$ 做 inverse Otsu 二值化得 $I$，连通域分析得 $\{C_j\}_{j=1}^N$；
- **Contour support band**: $B = \mathrm{Dilate}(M_c; K^{\text{ellipse}}_{3 \times 3})$；
- **Component-level overlap filtering**: 对每个 $C_j$ 计算 $o_j = |C_j \cap B|$, $r_j = o_j / |C_j|$，保留满足 $|C_j| \ge A_{\min}$ 且 $(o_j \ge P_{\min} \text{ or } r_j \ge R_{\min})$ 的连通域；
- **Final mask**: $M_o = \bigcup_{j \in \mathcal{K}} (C_j \cap B)$。

默认参数 $A_{\min}=8, P_{\min}=5, R_{\min}=0.03$。该模块**不含任何可学习参数**，可完全复现。

### 4. 自监督训练目标 (Self-Supervised Objective)

**NT-Xent 对比一致性损失** (over a mini-batch of $B$ images, $2B$ augmented samples):

$$\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2B} \mathbf{1}_{[k \ne i]} \exp(\mathrm{sim}(z_i, z_k) / \tau)}$$

$$\mathcal{L}_\mathrm{con} = \frac{1}{2B} \sum_{i=1}^{B} (\ell_{2i-1, 2i} + \ell_{2i, 2i-1})$$

**View-aligned Sobel 伪边重建损失** (对 $M_{c,1}, M_{c,2}$ 各一次):

$$e = \mathbf{1}\big[\sqrt{(\nabla_x x')^2 + (\nabla_y x')^2} > \delta\big], \quad e_i = v_i(e)$$

$$\mathcal{L}_\mathrm{pix} = \frac{1}{2}\sum_{i=1}^{2}\|M_{c,i} - e_i\|_1$$

$$\mathcal{L}_\mathrm{grad} = \frac{1}{2}\sum_{i=1}^{2}\big(\|\nabla_x M_{c,i} - \nabla_x x_i\|_2^2 + \|\nabla_y M_{c,i} - \nabla_y x_i\|_2^2\big)$$

$$\mathcal{L}_\mathrm{rec} = \mathcal{L}_\mathrm{pix} + \gamma\,\mathcal{L}_\mathrm{grad}$$

**总损失**:

$$\mathcal{L} = \lambda_1 \mathcal{L}_\mathrm{con} + \lambda_2 \mathcal{L}_\mathrm{rec}$$

默认超参数: $\tau = 0.1, \gamma = 0.5, \lambda_1 = 1.0, \lambda_2 = 0.8, \delta = 0.2$。

---

## 项目结构 / Repository Layout

```
dl_cv_paramnet/
├── README.md                     # 项目文档（本文件）
├── requirements.txt              # 依赖列表
├── LICENSE                       # MIT License
│
├── configs/                      # 配置文件
│   └── default.json
│
├── data/                         # 数据
│   ├── README.md                 # 数据准备说明
│   ├── deepjiandu/               # DeepJiandu 数据集挂载点（需自行下载，见下文）
│   ├── eval_subset_100/          # 100-image 评估子集
│   │   ├── image_ids.txt         # 子集图像 ID 列表（与论文 §4.1 对应）
│   │   ├── seed.txt              # 随机采样使用的 seed
│   │   └── ground_truth/         # 手工 mask GT（解压自 ground_truth.zip）
│   └── liye_qin/                 # 里耶秦简零样本迁移子集（20 张，见论文 §4.7.4）
│
├── src/                          # 源代码
│   ├── config.py
│   ├── losses.py
│   ├── data.py
│   ├── metrics.py                # BF-Score / IoU / Dice / MCC / Contour Recall
│   └── models/
│       ├── feature_extractor.py
│       ├── parameter_predictor.py
│       ├── cv_execution.py       # Canny + Closing
│       ├── overlap_mask.py       # 确定性 overlap-mask 生成 G
│       └── model.py
│
├── scripts/                      # 训练 / 评估 / 推理 脚本
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── ablation.py
│   ├── sensitivity.py            # 论文 §4.5.3 敏感性分析
│   ├── liye_transfer.py          # 论文 §4.7.4 零样本迁移
│   └── visualization.py
│
└── checkpoints/                  # 预训练权重存放目录
```

---

## 安装 / Installation

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10 (with CUDA ≥ 11.0 recommended)
- OpenCV ≥ 4.5

### 依赖安装

```bash
pip install -r requirements.txt
```

或者最小集合：

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm scipy scikit-image
```

---

## 数据准备 / Data Preparation

### DeepJiandu 主数据集

本方法使用公开的 **DeepJiandu** 数据集 (Liu et al., *Scientific Data*, 2025)。请从其官方页面下载并按以下结构放置：

```
data/deepjiandu/
├── train/      # 5,922 张训练图
├── val/        # 751 张验证图
└── test/       # 743 张测试图
```

本项目**不使用** DeepJiandu 的字符级或像素级标签训练 ParamNet，仅使用图像本身做自监督训练。

### 100-image 评估子集

论文 §4.1 中用于跨方法 mask 评估的 100 张图像子集 + 手工 ground-truth mask 提供在：

```
data/eval_subset_100/
├── image_ids.txt        # 从 DeepJiandu test split 随机采样的 image IDs
├── seed.txt             # 复现随机采样使用的 seed
└── ground_truth.zip     # 100 张二值 GT mask（PNG 格式 + JSON 元数据）
```

**解压使用**：

```bash
cd data/eval_subset_100
unzip ground_truth.zip -d ground_truth/
```

### 里耶秦简零样本迁移子集

论文 §4.7.4 中用于零样本泛化分析的 20 张里耶秦简放在：

```
data/liye_qin/
└── images/        # 20 张零样本测试图
```

注：该子集**仅用于零样本定性分析**，不含像素级 ground truth；论文中 99/121 的检测比是手工目测验证而非像素级指标。

---

## 使用方法 / Usage

下列命令均假定在 repo 根目录运行。

### 1. 训练 / Training

```bash
# 标准训练（论文默认设置：100 epochs, batch=128, lr=1e-4）
python scripts/train.py \
    --data_dir data/deepjiandu \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --device cuda

# 从断点恢复
python scripts/train.py --resume checkpoints/dl_cv_paramnet/best.pth

# 自定义配置
python scripts/train.py --config configs/default.json
```

### 2. 评估 / Evaluation

在 100-image 评估子集上复现论文 Table 3：

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/dl_cv_paramnet/best.pth \
    --eval_subset data/eval_subset_100 \
    --metrics bf_score iou dice mcc contour_recall \
    --output outputs/main_results.json
```

### 3. 消融实验 / Ablation Study

```bash
# 复现论文 §4.5.1 消融（5 种配置）
python scripts/ablation.py --output_dir experiments/ablation
```

### 4. 敏感性分析 / Sensitivity Analysis

```bash
# 复现论文 §4.5.3 (Tables 6 & 7)
python scripts/sensitivity.py --output_dir experiments/sensitivity
```

### 5. 里耶秦简零样本迁移 / Liye Qin Zero-Shot

```bash
# 复现论文 §4.7.4 Fig. 11
python scripts/liye_transfer.py \
    --checkpoint checkpoints/dl_cv_paramnet/best.pth \
    --input_dir data/liye_qin/images \
    --output_dir outputs/liye_transfer
```

### 6. 推理 / Inference

```bash
# 单图像
python scripts/inference.py \
    --checkpoint checkpoints/dl_cv_paramnet/best.pth \
    --input image.png \
    --output result_mask.png \
    --overlay  # 同时输出叠加可视化

# 批量
python scripts/inference.py \
    --checkpoint checkpoints/dl_cv_paramnet/best.pth \
    --input_dir images/ \
    --output_dir results/
```

---

## 评价指标 / Evaluation Metrics

论文采用 5 个互补指标，所有指标均在 $M_o$ (final overlap mask) 上计算：

| 指标 | 定义 | 角色 |
|------|------|------|
| **BF-Score** | $F_1 = \frac{2 P_b R_b}{P_b + R_b}$, with 2-pixel tolerance | Primary (边界匹配) |
| **IoU** | $\|M_p \cap M_g\| / \|M_p \cup M_g\|$ | Primary (区域重叠) |
| **Dice** | $2\|M_p \cap M_g\| / (\|M_p\| + \|M_g\|)$ | Primary (对小目标敏感) |
| **MCC** | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Auxiliary (前后景平衡) |
| **Contour Recall** | $\|\partial M_g \cap N_\epsilon(\partial M_p)\| / \|\partial M_g\|$ | Auxiliary (边界覆盖) |

字符级辅助分析使用中心匹配协议下的 macro/micro detection rate + center-based precision + F1 (论文 §4.6, Table 8)。

---

## 数据增强 / Data Augmentation

匹配灰度简牍扫描的真实变化，**仅使用几何与轻量光度扰动**，不引入噪声型增强：

- 随机旋转 $[-15^\circ, +15^\circ]$
- 随机缩放 $[0.9, 1.1]$
- 亮度/对比度抖动 $\pm 0.2$

不使用翻转（竹简文本是有方向语义的）。详见论文 §3.9。

---

## 配置 / Configuration

### Model Config

```python
@dataclass
class ModelConfig:
    feature_dim: int = 512
    predictor_hidden_dim: int = 256

    # 参数映射（与论文 §3.3 一致）
    # t_l = 100 * σ(p1)               -> [0, 100]
    # t_h = t_l + (200 - t_l) * σ(p2) -> [t_l, 200]
    # k_m = 10 * σ(p3) + 3            -> [3, 13]
    se_reduction: int = 16          # SE channel attention reduction ratio
    projection_dim: int = 128       # contrastive projection head output dim
```

### Training Config

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"

    # 损失权重（论文默认）
    lambda_contrastive: float = 1.0  # λ_1
    lambda_reconstruction: float = 0.8  # λ_2
    temperature: float = 0.1            # τ
    gamma: float = 0.5                  # γ
    sobel_threshold: float = 0.2        # δ
```

---

## 差分化技术细节 / Differentiable Approximations

为使梯度能从自监督目标回流到 $(t_l, t_h, k_m)$，训练期用如下软近似（评估期则使用标准 OpenCV 实现）：

**Canny 软阈值** (sigmoid-based)：

$$\widetilde{g} = \frac{g - t_l}{\max(t_h - t_l, \Delta)}, \quad \widetilde{E} = \sigma\big((g - t_l) / T\big) \cdot \sigma\big((\widetilde{g} - 0.3)\,T\big)$$

**Soft NMS**: 用 $3 \times 3$ max-pool 近似比较。

**Morphological closing soft surrogate** (softmax-weighted mixture over six kernels):

$$\widetilde{M}_c = \sum_{k \in \{3,5,7,9,11,13\}} w_k(k_m) M_c^{(k)}, \quad w_k(k_m) = \mathrm{softmax}_k\!\big(-|k_m - k|/\tau_m\big)$$

**STE side path**: $k_m$ 经 straight-through estimator 圆到最近奇数，**仅用于训练期诊断和评估期硬执行**，梯度路径完全由 softmax mixture 承担。

详见论文 §3.9 和 Algorithm 1。

---

## 引用 / Citation

```bibtex
@article{jishou2026paramnet,
  title  = {DL-CV ParamNet: Self-Supervised Parameterization of Classical CV for Jiandu Manuscript Mask Extraction},
  author = {Shikai Jishou},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/shikai666/dl_cv_paramnet}
}
```

> Replace with the venue/DOI once the paper is accepted.

---

## 许可证 / License

本仓库代码遵循 **MIT License**。所使用的 DeepJiandu 数据集请参阅其原始 license。

---

## 联系 / Contact

如有问题或建议，请在 GitHub 上提交 [Issue](https://github.com/shikai666/dl_cv_paramnet/issues)。
