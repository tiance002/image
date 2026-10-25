# 本轮产物、资源与复现入口

本轮基于本地历史v4@47999，官方ACDC train1600/val406，修正后的RGBv2配对数据，以及同一份YOLOv8n。所有数据、模型权重和运行环境的软件包均使用本地已有资源；另取得少量官方网页和评估源码文本。没有自动push、merge、部署或替换默认权重。

## 结果索引

- `P0_REPORT.md`：颜色链路、标注ID适配、双阈值协议、input/v4/guard重新评价。
- `TRAINING_REPORT.md`：原生GT损失、C0/C1预算、训练损失/梯度、检测/画质/效率/区间。
- `BASELINE_COMPARISON.md`：官方技术来源/许可证/预处理与本地实测；AdaIR/Histoformer缺资源标记BLOCKED。
- `FAILURE_ANALYSIS.md` / `failure_cases.json`：全部95个bus/train GT的逐方法证据、最多3例/类型、FP排序与NMS证据边界。
- `FINAL_DECISION.md`：以最终实际数值作出的保留/停止决定。
- `COLOR_AUDIT.md` / `color_audit.json`：历史实物重编码、5200新版JPEG检查、历史weather配方未溯源的限制。
- `padding_equivalence.json`：后评估所需的既有非整除尺寸残差修复；训练/检测整除尺寸的实际C0 EMA新旧前向逐位一致。
- `GT_READINESS.md` / `gt_loss_smoke.json` / `annotation_integrity.json`：真实train来源、类别映射、冻结反传、跨划分原始文件哈希。
- `metrics.json`：200/406与四天气逐类AP50、AP50:95、PR数组、runtime Recall/FP/OPR。
- `predictions/<方法>/ap001.json`、`runtime025.json`、`metadata.json`：完整逐图预测，固定资源哈希、依赖版本、全406初始推理计时；bus/train图含pre-NMS候选。
- `bootstrap_p0.json` / `bootstrap_gt.json`：1000次配对、按天气分层的图像bootstrap。`quality_bootstrap.json`另复用项目统计函数计算对齐合成图PSNR/SSIM差值区间。
- `image_quality.json`：像素对齐合成dev200逐图PSNR/SSIM；LPIPS BLOCKED，不用跨时刻真实天气参考图冒充对齐真值。
- `benchmark.json`：固定每天气前10张、5张warmup的完整读图至CPU检测框回传计时；不同于早期metadata中不含读图的耗时。
- `plots/`：标准AP50 PR曲线与训练损失/梯度图。
- `audit_history/`：**INVALIDATED**。保存未做正annotation ID适配前的临时汇总，不得用于结论；原预测未改变。

## 权重和训练原始证据

固定初始化为`runs/real_v4_allweather/latest.pt`，iter47999，不混用best@43999。C0/C1目录分别为`runs/gt_next_phase_20261025_C0/`、`runs/gt_next_phase_20261025_C1/`：

- `iter_049999.pt`为新增2000步，`iter_051999.pt`与`latest.pt`为新增4000步的同一组参数。不同文件名的torch ZIP字节哈希可能不同，不能由此认定张量不同。
- 本轮固定评价最终EMA@51999，没有根据验证分数选择2k/4k或延长训练。
- `train_log.csv`保存每步损失、A/B梯度、固定/实际任务权重、裁剪范数上界和全流SHA。
- `sample_stream.jsonl`保存采样图像ID/路径及逐步累计输入/GT字节哈希。
- `training_evidence.json`保存步数、耗时、冻结状态、初始model/EMA哈希及checkpoint SHA。
- `runs/gt_next_phase_20261025/comparison_plan.json`包含6407个输入文件和6个训练源码的执行时哈希；`calibration.json`保存8个固定train批次及无优化更新的校准证据。

两组完整输入/GT流SHA均为`777dc1d35fa7ef4a12bea0916fd45009f046079dae56aad0546ed4b6739534b7`。权重固定为0.009570541647407782，初始20%梯度比只是工程校准目标，500步warmup后固定，不随验证分数调节。

## 固定执行入口

在已经具备本地资源且目标目录为空的项目副本内，使用原有Python3.11.15、torch2.6.0+cu124、Ultralytics8.4.104、NumPy2.4.4：

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONIOENCODING='utf-8'
python scripts/audit_pair_color.py
python scripts/audit_pair_color.py --replay-inputs
python scripts/audit_pair_color.py --verify-fixed
python scripts/evaluate_next_phase.py --name input
python scripts/evaluate_next_phase.py --name v4 --checkpoint runs/real_v4_allweather/latest.pt
python scripts/evaluate_next_phase.py --name guard --checkpoint runs/confidence_v8_20261025_guard/latest.pt
python scripts/evaluate_next_phase.py --score input v4 guard
python scripts/analyze_next_phase_failures.py --methods input v4 guard
python scripts/bootstrap_next_phase.py --pairs v4:input guard:input --resamples 1000 --out results/next_phase/bootstrap_p0.json
python scripts/report_next_phase.py
python scripts/run_gt_comparison.py --prepare-only
python scripts/run_gt_comparison.py --calibration-only
python scripts/run_gt_comparison.py
python scripts/run_next_phase_evaluation.py
```

当前工作区已经完成相应目录创建，不能原地重新训练。训练入口拒绝覆盖已有实验；完整复跑应使用新的项目输出空间，或通过`--name`另建训练对照，并显式对应新权重和评估方法名。后评估默认入口锁定本轮canonical目录，不会自动选新目录中的“最优”权重。构建颜色数据也须显式选择另一新输出目录，见两个builder的`--help`；不覆盖历史数据。

Git源码副本保留源码、报告和聚合证据；数据、模型、原始预测/GT坐标与完整采样日志仍在本地科研工作区，不作为公开仓库自动上传。仅有Git源码无法直接复现科学数值，必须准备上述本地资源。执行时raw源码哈希包含换行字节；Git若规范为LF，按规范化源码哈希核对内容，再为新实验建立预注册清单。
