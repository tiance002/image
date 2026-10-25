# SafeRestore-RF

面向雨雾道路感知的任务感知图像复原研究：改善图像的同时，检查复原是否损伤下游目标检测。项目把逐类 AP、输入目标保留率与失败案例纳入候选筛选，关注“看起来更清晰”与“检测更可靠”之间的差距。

**当前研究状态（2026-10-25）：保留 v4 基线，停止本轮 C1 配方续训。** 已完成 RGB/评估正确性修复和真实 GT 原生 YOLO 损失的 C0/C1 公平对照，各新增 4000 步。C1 没有解决 bus/train 掉分，不替换默认权重。本轮只提交本地，不自动 push；仓库仍不附带数据或预训练模型。

## 方法与代码入口

当前实验主线采用 **NAFNet（17.06M 参数）** 复原网络，复用冻结 YOLOv8n 的特征和原生真实 GT 分类/box/DFL 损失。原有骨干和损失保留，新增两条同预算数据流；置信度保护为历史对照，本轮关闭。检测器参数冻结，复原图像分支仍保留梯度。置信度教师使用原始退化图像；按图像与类别选择高分锚点，惩罚学生类别概率下降。它不包含直接定位约束，也不保证最终 AP 或虚警改善。

| 入口 | 用途 |
|---|---|
| [`scripts/train_restorer.py`](scripts/train_restorer.py) | 当前 NAFNet 实验训练 |
| [`src/saferestore/losses/`](src/saferestore/losses/) | 检测特征损失与置信度保护 |
| [`scripts/run_confidence_comparison.py`](scripts/run_confidence_comparison.py) | 同检查点、同预算 control / guard 续训 |
| [`scripts/report_confidence_comparison.py`](scripts/report_confidence_comparison.py) | 从本地完整实验产物离线生成报告 |
| [`src/saferestore/train.py`](src/saferestore/train.py)、[`configs/baseline.yaml`](configs/baseline.yaml) | 早期双分支融合模型与工程冒烟入口 |

早期双分支、退化感知融合与界面演示仍保留为工程原型；它们与当前 NAFNet 主线的架构、训练入口和检查点不同，不能用其冒烟结果证明下表的科研结论。

## 下一阶段实测（标准AP conf=0.001）

AP使用COCO官方bbox匹配、正annotation ID、AP50/AP50:95/PR和maxDets100；runtime Recall/FP/OPR另跑conf=0.25、maxDet300。配置补齐train类。当前200/406均是val模型选择集，不是独立test。

| 方法 | 雨雾200 AP50 | 雨雾200 AP50:95 | 全天气406 AP50 |
|---|---:|---:|---:|
| input | 0.256338 | 0.155289 | 0.233510 |
| v4 | 0.251481 | 0.150662 | 0.220924 |
| guard | 0.251884 | 0.151387 | 0.224067 |
| C0 | 0.249156 | 0.145423 | 0.214648 |
| C1 | 0.252121 | 0.150022 | 0.215576 |

主200 bus AP50：C0 0.148824 → C1 0.148829（input 0.173570），95%差值区间跨零；bus/train运行时目标集合无新增恢复。主AP50的C1−C0区间跨零，AP50:95有局部正向信号；全天气train相对input下降7.132个百分点。合成对齐dev200 PSNR为C0 28.496dB、C1 28.430dB；LPIPS及AdaIR/Histoformer外部推理因缺本地资源BLOCKED。

保留正确性修复和GT研究接口，不把当前C1权重当作已验证改进。详见[最终决策](results/next_phase/FINAL_DECISION.md)、[训练与完整结果](results/next_phase/TRAINING_REPORT.md)、[成熟模型核查](results/next_phase/BASELINE_COMPARISON.md)及[产物/复现说明](results/next_phase/ARTIFACTS.md)。本轮27项相关测试通过；源码快照的聚合范围见[SNAPSHOT_SCOPE](results/next_phase/SNAPSHOT_SCOPE.md)。

新增入口：`scripts/run_gt_comparison.py`、`scripts/run_next_phase_evaluation.py`。固定Python3.11.15、torch2.6.0+cu124、Ultralytics8.4.104、NumPy2.4.4；其余实际版本记录于execution_manifest。旧数据存在颜色缺陷，不应按旧命令重建替代数据后声称重现历史分数。

## 历史置信度结果（旧 conf=0.25 截断 AP）

主口径为 **ACDC val 中固定的 200 张雨雾图像（fog100 + rain100）**，补充口径为同一 val 的 406 张全天气图像。两者均已用于模型选择，**不是独立 test**。指标为 person、bicycle、car、motorcycle、bus、train、truck 七类等权 mAP50。

| 口径 | 输入原图 | v4@48k | control@52k | guard@52k |
|---|---:|---:|---:|---:|
| 雨雾 200 张 mAP50 | 0.204221 | 0.203157 | 0.192734 | 0.200764 |
| 全天气 406 张 mAP50 | 0.187137 | 0.179454 | 0.177358 | 0.182848 |

control 与 guard 从同一 v4 检查点出发，各续训 4000 步，使用 seed42；完整批次轨迹、教师选择和冻结资源哈希已核对一致。guard 的主口径增量为 +0.008030，其中约 94% 的宏平均增量来自 train：其 AP50 从 control 的 0.257426 恢复至 0.310231。去除 train 后，六类均值增量仅 +0.000568。

bus AP50 由输入的 0.163538 降至 guard 的 0.108454，下降 0.055084，超过单类 0.05 筛选红线。因此拒绝 guard，继续保留 v4；这也不表示 v4 已通过完整安全验收。主口径 motorcycle 在固定基线中的 AP50 为零，相关解释仅适用于该检测器与该集合。

完整逐类、逐天气、虚警、OPR 与对称丢失/恢复结果见 [`results/confidence_v8_20261025/REPORT.md`](results/confidence_v8_20261025/REPORT.md)；实验定义与局限见[置信度对照记录](docs/CONFIDENCE_COMPARISON_2026-10-25.md)和[当前状态](docs/STATUS.md)。该历史置信度阶段没有三种子验证、置信区间、显著性检验或独立 test；PSNR/LPIPS 红线未验证，不宣称完整 D1 通过或 SOTA。

## 安装与本地检查

需要 Python 3.10 或更新版本。以下命令供用户在自己的环境执行；安装依赖可能需要网络。请按本机平台选择兼容的 PyTorch / CUDA 环境。

```bash
git clone https://github.com/tiance002/image.git
cd image
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活：

```bash
source .venv/bin/activate
```

在仓库根目录安装项目与可选依赖：

```bash
python -m pip install -e '.[dev,metrics,detection,demo]'
python scripts/validate_project.py
python -m saferestore.train --config configs/baseline.yaml --smoke-test
python -m pytest
```

项目结构检查需要 PyYAML；合成张量冒烟需要 PyTorch 等基本依赖，但均不需要数据集或模型权重。冒烟只验证早期工程链路的两步优化，不能复现科研指标。已记录的历史验证为 **105 项核心测试通过**；当时未安装 Gradio，`test_app_runtime.py`、`test_precision_benchmark.py`、`test_video_benchmark.py` 三个 UI 测试文件未执行，不能计为通过。完整环境可按上述命令运行全部测试；仅核查该核心范围可执行：

```bash
python -m pytest --ignore=tests/test_app_runtime.py --ignore=tests/test_precision_benchmark.py --ignore=tests/test_video_benchmark.py
```

## 运行置信度公平对照

**全新克隆不能直接复现上述科学数值。** 仓库不附带 ACDC 数据、标注、YOLO 权重或 v4 复原检查点，也不提供数据或模型下载步骤。真实训练要求本地资源齐备；缺少权重时应立即报错，不以随机权重代替。请先合法获取资源，并依照[数据指南](docs/DATA_GUIDE.md)、配置与脚本参数准备本地清单和路径。

默认对照入口要求以下资源，并要求清单中的图像/目标路径在本机有效。`latest.pt` 必须对应 v4 已完成 48000 步的检查点，含恢复训练所需状态：

```text
runs/real_v4_allweather/latest.pt
yolov8n.pt
data/real_weather/manifests/train.csv
data/real_scene/manifests/train.csv
runs/acdc_benchmark/evaluation/weather_input_all406.json
```

除数据与检查点外，报告生成还依赖输入原图、v4 以及两个续训组的完整本地预测和对应标注记录。运行前核对 [`configs/detector.yaml`](configs/detector.yaml) 与脚本中的固定评测协议。采用新的实验名，避免覆盖已有产物：

```bash
python scripts/run_confidence_comparison.py --name my_confidence_trial
python scripts/report_confidence_comparison.py --comparison my_confidence_trial
```

当前顺序执行器使用 Windows 风格的 `PYTHONPATH` 分隔符；在 Linux / macOS 运行此实验前需将其调整为本机 `os.pathsep`。安装和基础检查命令不受此限制。

默认每组新增 4000 步，学习率总日程为 64000 步；control 的新增置信度权重为 0，guard 为 10。输出写入 `runs/my_confidence_trial*`；已有同名目录会被拒绝。报告命令读取已完成的实验，不启动训练或图像推理；缺少完整证据时不能仅用公开汇总结果重建 AP。

## 公开内容与使用边界

- `src/`、`scripts/`、`tests/`、`configs/`：实现、检查与实验入口。
- `docs/`：研究协议、数据说明、验收规则与当前结论。
- `results/`：聚合报告、配置、训练曲线、哈希和公平性记录。逐图预测与 GT 坐标有意未上传；重新计算 AP 需要自行在本地准备这些资源。

公开副本经过路径脱敏，并省略私人调度脚本与工作记录；实验原始源码哈希对应当时的本地快照，不能将公开副本冒充同一冻结快照。当前结果只支持固定检测器与验证集上的探索性判断，不构成自动驾驶安全认证。

NAFNet、Ultralytics、ACDC 数据与预训练权重分别遵循原作者的许可和使用条件。公开仓库没有为这些第三方资源重新授权，也未额外选定覆盖全部内容的新许可证；详情见 [`PUBLICATION_NOTES.md`](PUBLICATION_NOTES.md)。

