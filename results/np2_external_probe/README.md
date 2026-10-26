# NP2 外部基线探针：工件清单

本目录是一次性对比实验的完整可复现记录。**不含权重、不含复原后的图片**（体积原因），
但每条结论都能用下面的命令 + 公开可校验权重重跑出来。

## 交付文档（先读这三份）

| 文件 | 内容 |
|---|---|
| `DECISION.md` | 六个必答问题的结论、未完成项、限制、下一轮唯一主实验建议 |
| `EXTERNAL_BASELINES.md` | 官方预训练模型（Histoformer / AdaIR）的来源、版本、SHA256、推理适配、逐项指标 |
| `SUPERVISION_AND_SCALE_PROBE.md` | bus/train 正样本监督覆盖 + 尺度顺序（先缩放 vs 先复原）只读诊断 |

## 数据与统计

| 文件 | 内容 |
|---|---|
| `metrics.json` | input / v4 / C0 / C1 / histoformer_real 在 rain、snow、all406、rain_fog200 上的 mAP50、mAP50:95、逐类 AP、Recall(.25)、FP/图、OPR |
| `bootstrap.json` | 配对图像 bootstrap（1000 次，seed 20261025）的差值区间 |
| `bus_train_gt_matrix.csv` | rain+snow 的 63 个 bus/train GT，逐 GT 列出 5 个方法的 hit/miss（同一目标同一行） |
| `quality_lpips.json` | 合成对齐 dev 上的 PSNR / SSIM / LPIPS（含外部模型） |
| `supervision_probe.json` | 64 张固定样本的逐类 GT 数、正 anchor、零正比例、权重总量、cls/box/DFL 分解、梯度范数 |
| `scale_order_probe.json` | 24 张上路径 A/B 的逐 GT 匹配、IoU、分数、原生检测损失 |
| `tiling_consistency.json` | 分块推理与全幅推理的一致性（PSNR / 接缝差） |
| `external_sources.json` | 权重与源码的官方地址、commit、SHA256、依赖版本、AdaIR 阻断证据 |
| `predictions/histoformer_real/` | `ap001.json`（conf=0.001）与 `runtime025.json`（conf=0.25），与 `results/next_phase/predictions/` 同构 |

## 未纳入版本库的产物（本地保留）

- `restored/histoformer_real/`（488 MB，200 张 ACDC rain/snow 复原 PNG）
- `restored/histoformer_dev/`（134 MB，200 张合成对齐 dev 复原图）
- `third_party/weights/histoformer_official_net_g_real.pth`（66,603,665 字节，
  SHA256 `8ae731d0fd5e2761cec7428322c7602146cba42c73188cc4b0fcc17232065958`）

## 重跑顺序

```bash
# 1) 复原（官方权重；640/64 分块、FP32）
python scripts/np2_histoformer_restore.py \
  --weights third_party/weights/histoformer_official_net_g_real.pth \
  --name histoformer_real --weather rain snow

# 2) 冻结检测（不 import einops/basicsr，避免污染检测栈）
python scripts/np2_detect.py --restored-root results/np2_external_probe/restored \
  --name histoformer_real --weather rain snow

# 3) 汇总指标 + 相对 input 的 GT 级恢复/丢失
python scripts/np2_report_metrics.py --external histoformer_real

# 4) 区间估计与 GT 对照表
python scripts/np2_bootstrap.py
python scripts/np2_gt_matrix.py

# 5) 画质（合成对齐 dev；LPIPS 装在 third_party/pylibs）
python scripts/np2_histoformer_dev.py \
  --weights third_party/weights/histoformer_official_net_g_real.pth --name histoformer_dev
python scripts/np2_quality_lpips.py --methods input v4 C0 C1 histoformer_real \
  --restored-root results/np2_external_probe/restored/histoformer_dev

# 6) 两项只读诊断
python scripts/np2_supervision_probe.py --grad --batch 4 --grad-batch 1
python scripts/np2_scale_order_probe.py
```

`third_party/` 与 `var/` 都在 `.gitignore` 里；权重与临时日志不入库。
