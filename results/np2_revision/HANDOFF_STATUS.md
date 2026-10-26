# NP2-R 交接状态（未完成，供接手者继续）

生成时间：2026-10-26 13:55
工作区：`D:/数字图像处理/SafeRestore-RF`（真实科研工作区，含权重/数据/预测）
审查基线：`tiance002/image` @ `next-phase-gt-20261025` / `3d53ca0`
本轮产出目录：`results/np2_revision/`（未覆盖 `results/np2_external_probe/`）

> 本文件是**进度交接**，不是交付物。任务书要求的 `REVISION_REPORT.md` **尚未撰写**，
> 因为其中一半数字依赖尚未跑完的重推理（见第三节）。不要把本文件当最终报告引用。

---

## 一、总体状态

| 工作项 | 代码 | 验证 | 全量结果 |
|---|---|---|---|
| Histoformer 分块融合修复 | 已改 | **已通过**（恒等模型逐像素 0 差） | **未跑**（200 张重推理只完成 10 张，已清理） |
| 逐类监督诊断修复 | 已重写 | **已通过**（7/7 针对性测试） | **未跑**（64 张 × 3 检查点未执行） |
| 尺度对照修复 | **未动** | — | — |
| GT2730 矛盾裁定 | 已查清 | 已完成 | 已确认（见第四节） |
| REVISION_REPORT.md | 未写 | — | — |

---

## 二、已完成并验证的部分

### 2.1 Histoformer 分块融合（`scripts/np2_histoformer_restore.py`）

**缺陷**：一维 Hann 窗边界为 0，`clamp_min(1e-6)` 后两维相乘，角点权重约 `1e-12`；
融合又把分母截断到 `max(weight, 1e-8)`。等于把 `1e-12` 的权重除以 `1e-8`，
输出被缩小 1e-4，恒等模型四角变黑。

**修复**：
1. 窗口改为 `hann_window(size + 2)[1:-1]`——去掉零端点的汉宁窗，任何位置权重严格 > 0；
2. 分子 `sum(out * w)` 与分母 `sum(w)` 用**同一个** `w`；
3. 分母**不再截断**，除法逐像素精确；
4. 增加全覆盖断言：任何像素权重 ≤ 0 直接抛错（防止块位置遗漏被静默掩盖）。

**验证证据**（`results/np2_revision/fusion_identity_check.json`）：

| 检查 | 结果 |
|---|---|
| 恒等模型，12 个用例（常量图/随机图 × 1920×1080、1080×1920、700×500、641×961、640×640、100×130） | `max_abs_diff = 0.0`，全部通过 |
| 四角最大绝对差 | `0.0`（旧版角点比值 = `1e-4`，即损失 99.99% 亮度） |
| 旧公式复现（常量 1080×1920） | 角点 `0.0001`（应为 0.5），近黑像素占比 `0.01%` |
| 真实权重抽检（2 张 rain） | 旧输出角点 `[0,0,0]`，新输出 `[208,210,214]`，新旧最大差 236 |

**未完成**：200 张（rain 100 + snow 100）重推理只跑了 10 张（约 15.7 s/图，全程约 52 分钟），
为避免"10 张被误当成 200 张"已删除半成品目录。

### 2.2 逐类监督诊断（新文件 `scripts/np2r_supervision_probe.py`）

旧文件 `scripts/np2_supervision_probe.py` **保持不变**（历史证据保留），修订版另存。

修的三处：

**A. 坐标尺度**：`pred_bboxes`/`anchor_points` 是**网格坐标**（`bbox_decode` 直接产出），
`target_bboxes` 是**像素坐标**（assigner 返回）。原生 `BboxLoss` 在调用前做
`target_bboxes / stride_tensor`。旧版直接拿两者算 IoU，不同 stride 层被缩放的比例不同，
IoU 与 `bbox2dist` 全部失真。

**B. 权重形状**：原生是 `weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)` → `[N,1]`；
旧版用 `[N]`，`[N,1] * [N]` 广播成 `[N,N]`。`DFLoss.__call__` 返回的也是 `[N,1]`（keepdim），
同样中招。

**C. 校验范围**：现在**逐批**校验两件事——
- 七类分量之和 == 原生总损失（同 gain、同 `target_scores_sum`、同 batch 口径）；
- 七类梯度向量之和 == 原生总损失的梯度向量（比向量本身：相对 L2 误差、余弦、最大元素差）。

**针对性测试**（`scripts/np2r_probe_tests.py`，CPU，7/7 通过）：

| 用例 | 结果 |
|---|---|
| zero_positive 零正 anchor | PASS |
| single_positive 单正 anchor | PASS |
| multi_positive 多正 / 跨类别 | PASS |
| multi_stride 8/16/32 三层混合 | PASS |
| crowd_rider_mask 忽略掩码（正监督保留、负监督去除） | PASS |
| **反例：不除 stride**（必须失败） | 原生 7.217 vs 反例 13.441（+86%） |
| **反例：weight=[N] 广播**（必须失败） | 原生 7.217 vs 反例 53.764（+644%） |

两个反例的数字就是"最小复现"：缺陷一旦写错，损失会被放大近 1 倍到 6 倍。

**未完成**：64 张 × v4/C0/C1 的统计与逐批梯度校验**没有跑**。

---

## 三、完全没做的部分

### 3.1 `scripts/np2_scale_order_probe.py`（未改动任何一行）

现状仍是旧逻辑，`REVISION_REPORT.md` 第 5 节需要的内容全部缺失：
- 路径 A 直接送浮点复原结果；路径 B 多一次 `uint8` 量化；
- A 复原的是含 114 灰边的 640 画布，B 是全分辨率复原后再 pad；
- 没有 NMS 后 `conf=.25` 的检出/丢失/新增 FP，没有同协议 AP；
- 没有直接输入对照；
- 没有按真实 GT 尺寸分桶；
- 两个零正 bus 的"候选中心是否落入框内"没有算，"小于 8 像素必然无 anchor"的错误概括仍在报告里。

建议的改法（下一步实现时照此做）：
1. 写一个 `letterbox_float()`，几何参数与 `letterbox_rgb` 完全一致（同 `scale`/`round`/`left,top`/`INTER_LINEAR`），但全程 float32、灰边填 `114/255`；
2. 四条管线共用同一检测器预处理：`A`（letterbox→复原→检测）、`A_nopad`（resize→复原→pad→检测，历史管线对照）、`B`（全分辨率复原→letterbox→检测）、`D`（letterbox→直接检测，直接输入对照）；
3. 量化只在需要时对**两边同位置**做（640 画布 → uint8 → float）；
4. 分块上下文差异单独记录（A 复原 640 单块；B 复原 1920×1080，tile=1024/overlap=64 共 4 块），并说明它不是被隔离掉的变量；
5. 补 NMS（IoU .7 / maxDet 300 / 同类别映射）后的 `conf=.25` 检出、丢失、新增 FP，以及该 24 张子集上的同协议 AP；
6. 零正 bus 逐 stride 级统计：anchor 中心落在 GT 框内的数量（stride 8 的 anchor 中心在 `4,12,20,...`，8×18 像素的框**完全可能**包含中心，必须实算）。

### 3.2 结论修订（未写进报告）

以下说法必须修订或撤回，逐条对应：

| 旧说法 | 问题 | 处理 |
|---|---|---|
| "bus/train 梯度占 15%–16%，监督已充分" | 梯度范数占比不可加和；且"诊断样本有监督"≠"整个训练期监督充分" | 改名为"梯度范数占比（不可加和）"，加限定 |
| "已经排除监督覆盖问题" | 只覆盖 64 张 train 帧，不能排除全训练期 | 撤回 |
| "已证明主要是小目标被抹掉" | bicycle 是小目标类别，但未按真实 GT 尺寸分桶 | 降级为待验证 |
| "一个外部检查点近似恒等，所以排除更强预训练复原路线" | 单检查点单任务，不能外推 | 撤回 |
| "区间跨零，所以证明不更差" | 跨零只能说明未检出差异，不是等价性证明 | 撤回 |
| "裁剪 C2 失败，所以尺度顺序不是主因" | C2 未启用 GT 任务梯度、且是裁剪采样而非路径对齐 | 撤回 |

### 3.3 旧 Histoformer 数字必须标为待修订

`results/np2_external_probe/` 里 `histoformer_real` 的 rain/snow 200 张指标**全部受融合缺陷影响**
（四角近黑）。在新结果出来前，任何引用都应注明"融合缺陷版，待重算"。
`metrics.json`、`EXTERNAL_BASELINES.md` 第 5 节、`DECISION.md` 中相关数字都属于这一类。

---

## 四、GT2730 矛盾：已裁定

`EXTERNAL_BASELINES.md` 第 145 行说 snow GT 2730 在 v4 上被检出，
而 `bus_train_gt_matrix.csv` 第 63 行写 v4 为 `miss`。两者都不算错，但**协议不同且原文没写协议**：

| 协议 | v4 是否命中 GT 2730 | snow 全部 bus 命中 |
|---|---|---|
| AP 协议 conf=0.001 / maxDet=100 | **是** | 2730, 2757 |
| 运行时协议 conf=0.25 / maxDet=300 | **否**（v4 在 snow 上一个 bus 都没检出） | 无 |

CSV 用的是运行时协议，所以写 `miss` 是对的。报告那句话是从 conf=.001 的匹配里摘的，
**必须补上协议限定**，否则会被读成矛盾。修订时应写成：
"snow GT 2730 在 AP 协议（conf=.001）下被 v4 命中，在运行时协议（conf=.25）下所有方法均未命中"。

---

## 五、接手后的执行顺序（可直接复制）

全部在 `D:/数字图像处理/SafeRestore-RF` 下执行，Python 用
`D:/Drivers/anaconda/envs/yolo_daily/python.exe`，并设 `PYTHONPATH="third_party/pylibs;src;."`。

```bash
# 1) 200 张重推理（约 52 分钟，GPU 独占，建议首件事）
python scripts/np2_histoformer_restore.py \
  --weights third_party/weights/histoformer_official_net_g_real.pth \
  --name histoformer_real_fusionfix --weather rain snow

# 2) 冻结检测（复用同一协议，不要动 input/v4/C0/C1 的既有预测）
#    注意：--out-root 必须显式改写，默认值会写进 np2_external_probe/predictions 覆盖旧结果
python scripts/np2_detect.py --name histoformer_real_fusionfix \
  --images results/np2_revision/restored/histoformer_real_fusionfix \
  --out-root results/np2_revision/predictions

# 3) 逐类监督诊断（64 张 × 3 检查点，含逐批梯度向量校验）
python scripts/np2r_supervision_probe.py --grad --batch 4 --grad-batch 1

# 4) 修好 np2_scale_order_probe.py 后再跑（见 3.1 的六条要求）
python scripts/np2_scale_order_probe.py

# 5) 汇总指标（需要把外部方法指向新目录，注意别覆盖 np2_external_probe 的 metrics.json）
```

已知的执行注意点：
- `np2_histoformer_restore.restore_tiled()` 现在返回 **3 个值**（`restored, n_tiles, min_weight`）。
  `np2_scale_order_probe.py` 目前用的是 `scripts/train_restorer.py` 里的同名 NAFNet 版本
  （仍返回单个张量，未受影响）；但如果改成复用 Histoformer 版本，必须同步改解包。
- `np2_detect.py` 的参数是 `--images`（不是 `--restored`），且 `--out-root` 默认指向
  `results/np2_external_probe/predictions`——直接跑会覆盖旧预测，务必显式指定新目录。
- 梯度诊断单图约需 8 次反传（1 次原生 + 7 个类别，`retain_graph=True` 共用一次前向），
  8GB 显存下 `--grad-batch 1` 是安全值；batch 4 会 OOM。

---

## 六、环境坑（会直接影响结果可信度）

1. **代理端口每次会话都变**：`git`/`curl` 走 `https_proxy` 时不要硬编码端口，
   直接读环境变量或 `netstat -ano | grep LISTENING`。本会话见过 63197 / 60660 / 59304 三个值。
2. **`cv2.imwrite` 在中文绝对路径下静默失败**，统一用 `imencode` + `write_bytes`。
3. **统计"某 GT 是否被检出"必须写明协议**（`.001` 还是 `.25`），这是第四节矛盾的根源。
4. **沙箱内改工作区文件不落盘**：所有会写文件的命令带 `dangerouslyDisableSandbox: true`，
   并在同一条命令里完成"写 + 验证"。
5. 用 Python 脚本改写源码时显式 `newline='\n'`，否则 Windows 会把整文件转成 CRLF。

---

## 七、文件清单

**新增**
- `scripts/np2r_supervision_probe.py` — 监督诊断修订版（含逐批 loss 与梯度向量校验）
- `scripts/np2r_probe_tests.py` — 7 个针对性测试（含 2 个必须失败的反例）
- `scripts/np2_fusion_identity_check.py` — 融合恒等验收 + 旧公式最小复现

**修改**
- `scripts/np2_histoformer_restore.py` — 融合归一化修复；`changed_frac_002` 改名为
  `changed_frac_maxchan_gt_0p02`（语义：三通道绝对差取最大后 >0.02 的像素占比）；
  默认输出目录改为 `results/np2_revision/restored`

**未动**
- `scripts/np2_supervision_probe.py`（旧版，保留作历史证据）
- `scripts/np2_scale_order_probe.py`（待重写）
- `results/np2_external_probe/` 全部文件

**产物**
- `results/np2_revision/fusion_identity_check.json` — 已完成，可信
- `results/np2_revision/restored/` — 空（半成品已清理）
