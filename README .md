# DL-CV ParamNet

**A Contrastive Self-Supervised Framework for Deep-Learning-Controlled CV Parameterization in Liye Qin Bamboo-Slip Contour Extraction**

本仓库提供论文 *DL-CV ParamNet* 的训练代码、推理代码、评测脚本、以及复现 100 张 DeepJiandu 主表数字、935/1298 字符覆盖率、20 张 Liye Qin zero-shot probe 所需的全部数据。

---

## 论文 vs 仓库对应关系

| 论文位置 | 实现文件 |
|---|---|
| §3.2 Preprocessing（in-graph HistEq 可微层） | `src/models/feature_extractor.py: class HistogramEqualization` |
| §3.3 Parameter Prediction Network | `src/models/parameter_predictor.py` |
| §3.4 Computer Vision Execution Module | `src/models/cv_execution.py` |
| §3.5 Overlap-Mask Generation $\mathcal{G}$ | **`contour_guided_ink_mask.py`** |
| §3.6 NT-Xent Contrastive Loss | `src/losses.py: class NTXentLoss` |
| §3.7 Pseudo-Edge Reconstruction Loss | `src/losses.py: class ReconstructionLoss` |
| §3.9 Training recipe & differentiable approximations | `scripts/train.py` + `src/losses.py` |
| §4.1 100-image evaluation subset | `data/deepjiandu_eval_ids.json` + `data/gt_masks/` |
| §4.3 Baselines | `scripts/evaluate.py`, `batch_canny_baseline.py`, 等 |
| §4.6 Character-level localization analysis | `data/character_coverage_deepjiandu.xlsx` |
| §4.7.3 Liye Qin zero-shot probe | `data/liye_subset.json`, `data/character_coverage_liye.xlsx` |
| Table 3 paired 95% CIs | `data/paired_statistics_full.csv` |

---

## 关键设计要点（影响复现）

### 1. HistEq 在模型内部，不在 dataloader 端

论文 §3.2 描述的 $x'=\mathrm{HistEq}(x)$ 是在 `src/models/feature_extractor.py` 的 `HistogramEqualization(nn.Module)` 里以**可微 rank-based CDF 形式**在模型 forward 第一层执行的（`forward()` 行 208-209），而**不是**预先在 CPU 用 `cv2.equalizeHist` 离线写盘。

- `configs/default.json` 里的 `"use_hist_eq": true` 默认开。
- 这意味着 `scripts/inference.py` 直接读原图喂模型即可——模型会自动做 HistEq，不需要任何 CPU 预处理步骤。
- 本仓库**完全不使用** `cv2.createCLAHE`（CLAHE 是局部 tile + 对比度裁剪的不同算子；论文 §3.2 明确不使用）。

### 2. 评测 mask 是三步生成的，不是 inference 一步出

论文报的 IoU/Dice/BF-Score/MCC 那批主表数字 **不是** `inference.py` 直接的输出，而是要再经过 `contour_guided_ink_mask.py` 做 overlap-mask 生成（§3.5）后才生成最终评测 mask $M_o$。完整复现流程见下文。

### 3. 训练超参数（**所有以代码为准**）

| 项 | 值 | 出处 |
|---|---|---|
| Optimizer | Adam | `configs/default.json` |
| Initial LR | $10^{-4}$ | 同上 |
| Weight decay | $10^{-4}$ | 同上 |
| Scheduler | Cosine annealing | 同上 |
| **Batch size** | **32** | 同上 |
| Epochs | 100 | 同上 |
| NT-Xent temperature $\tau$ | 0.1 | 同上 |
| Reconstruction weight $\lambda_2$ | 0.8 | 同上 |
| Gradient-consistency $\gamma$ | 0.5 | 同上 |
| Sobel threshold $\delta$ | 0.2 | 同上 |
| Rotation range | $[-15^\circ, +15^\circ]$ | `src/data.py: GeometricTransform` |
| **Scale range** | **$[0.8, 1.2]$** | 同上 |
| Horizontal flip prob | 0.5 | 同上 |
| Photometric jitter | 不使用 | 同上 |
| Canny $t_l$ 上界 | 100 | `src/models/parameter_predictor.py` |
| Canny $t_h$ 上界 | 200（相对参数化：$t_h=t_l+(t_{\max}-t_l)\sigma(p_2)$） | 同上 |
| 形态学闭运算核 $k_m$ | $\{3,5,7,9,11,13\}$ | 同上 |

---

## 三步复现流程

### 环境

```bash
pip install -r requirements.txt
```

要求：Python ≥ 3.8、PyTorch ≥ 1.10、OpenCV、scipy、scikit-learn、numpy、tqdm。

### 步骤 1：训练（或下载预训练 checkpoint）

```bash
# 从头训练（约 2 小时，RTX 3090）
python run.py train --config configs/default.json

# 或者直接用 checkpoints/best.pth（如附带）
```

`scripts/train.py` 自动读取 `data/processed/train/` 下的训练图（DeepJiandu 5,922 张训练集）。生成的 checkpoint 存在 `checkpoints/best.pth`。

### 步骤 2：模型推理 → 生成 raw contour PNG

```bash
python run.py inference \
    --checkpoint checkpoints/best.pth \
    --input data/processed/test/ \
    --output outputs/raw_contours/
```

`scripts/inference.py` 把每张测试图（原始灰度图）喂给模型，输出 `<id>_raw_contour.png`（即论文里的 $M_c$）。**模型 forward 内部自动做 HistEq**，无需在此阶段做额外预处理。

### 步骤 3：overlap-mask 生成（§3.5）→ 最终评测 mask

```bash
python contour_guided_ink_mask.py \
    --image_dir data/processed/test \
    --contour_dir outputs/raw_contours \
    --output_dir outputs/overlap_masks
```

`contour_guided_ink_mask.py` 实现论文 §3.5 的 overlap-mask 生成：
- inverse Otsu 在**原图** $x$ 上提 loose ink candidates（$A_\min=8$）
- raw contour PNG 经 $3\times 3$ 椭圆核 dilation 形成 support band $B$
- 保留满足 $o_j\ge P_\min=5$ **或** $r_j\ge R_\min=0.03$ 的 connected components
- 输出 $M_o=\bigcup_{j\in\mathcal{K}}(C_j\cap B)$ 为 `<id>_overlap_mask.png`

### 步骤 4：评测

把 `outputs/overlap_masks/` 跟 `data/gt_masks/` 配对，调 `scripts/evaluate.py` 计算 BF-Score/IoU/Dice/MCC/Contour Recall。这些就是论文 Table 2 报的数字。

---

## 数据 artifacts（位于 `data/`）

| 文件 | 内容 | 论文引用 |
|---|---|---|
| `data/deepjiandu_eval_ids.json` | 100 张 DeepJiandu 测试图的 ID 列表（随机抽样、fixed seed） | §4.1 |
| `data/gt_masks/<id>.png` | 100 张人工标注的 binary foreground mask GT，与上表 ID 一一对应 | §4.1 |
| `data/character_coverage_deepjiandu.xlsx` | 100 张图的 per-slip hand-counting sheet（M_i, K_i, 状态、备注），合计 $K/M=935/1298\approx 72.0\%$ | §4.6 |
| `data/liye_subset.json` | 20 张 Liye Qin slip 的 ID 列表、入选标准、与每张的 $K_i/M_i$ | §4.7.3 |
| `data/character_coverage_liye.xlsx` | 20 张 Liye 的 per-slip hand-counting sheet，合计 $K/M=99/121\approx 81.8\%$ | §4.7.3 |
| `data/paired_statistics_full.csv` | Table 3 完整 27 组精确数据：每行包含 $\Delta$、95% bootstrap CI（10000 重抽，percentile）、Holm-Bonferroni $p$、原始 Wilcoxon $p$ | Table 3 |

### `paired_statistics_full.csv` 字段说明

| 列 | 含义 |
|---|---|
| baseline / metric | 9 个 baseline × 3 primary metrics (BF/IoU/Dice) |
| n_images | 配对样本数（=100） |
| mean_ours / mean_baseline / delta_mean | 100 张的均值与配对差值 $\Delta=$ Ours − Baseline |
| ci95_lo / ci95_hi | 95% paired bootstrap CI 下界 / 上界 |
| ci_excludes_zero | CI 区间是否不含 0（"yes"=显著） |
| p_holm_bonferroni | 跨 27 组 Holm–Bonferroni 校正后的 Wilcoxon $p$ |
| wilcoxon_W / wilcoxon_p_raw | 原始 Wilcoxon 检验统计量和未校正 $p$ |
| note | 若该行 baseline 的 run 配置与论文 Table 1 不一致，此处会标记（详见下文） |

### Liye 选择标准（在跑模型之前确定，**不**根据模型输出筛选）

`data/liye_subset.json` 中已列出 3 条入选标准：

1. **C1**: 竹简扫描完整，无大段物理破损导致一连串字符位置缺失；
2. **C2**: 至少 3 个字符肉眼可识别（避免单字主导）；
3. **C3**: 至少含以下基底退化模式之一——明显木纹、水渍、薄墨/部分褪色。

筛选只看竹简物理状态，不看模型预测——单张覆盖率不参与入选决策。这是 §4.7.3 强调 "curated rather than random" 的具体含义。

---

## 已知与论文 Table 1 / 2 微小不一致

`data/paired_statistics_full.csv` 末列 `note` 中标记了两类 baseline 行，需要在正式投稿前用 canonical val-tuned per-image CSV 重跑：

- **CAE**: 仓库随附的 per-image 输出对应 `CAE_p75`（即 percentile=75 阈值）；论文 Table 1 报的 val-tuned configuration 是 q=80 in all folds。两者均值差异约 0.015 (BF)。
- **DexiNed**: 仓库随附的 per-image 输出是 fixed-threshold default run；Table 1 报的 val-tuned configuration 是 $\theta_e=0.6$ in 4 folds, $0.7$ in 1 fold。两者均值差异约 0.008 (BF)。

其他 7 个 baseline (Otsu, Sauvola w15, Fixed Canny, Val-tuned Canny, TEED, U-Net, DeepLabV3+) 的 per-image 输出与 Table 2 一致，CI 可直接用。

---

## 引用

```bibtex
@article{...,
  title={DL-CV ParamNet: A Contrastive Self-Supervised Framework for Deep-Learning-Controlled CV Parameterization in Liye Qin Bamboo-Slip Contour Extraction},
  author={...},
  journal={...},
  year={2026}
}
```

---

## 联系方式

如有任何复现问题，请在 GitHub 上提 issue，或邮件联系作者。
