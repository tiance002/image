> 此文件保留阶段性研究记录；当前结论、验证集口径和公开资源范围以根目录 README 与置信度公平对照报告为准。

# ACDC 检测标注接入与审计

更新时间：2026-10-05（Asia/Shanghai）

## 已完成

- 已从 ACDC 官方下载 `gt_detection_trainval.zip`，文件大小 4,173,937 字节；
- SHA-256：`9e9b3157c3e9d1de5bcf492ccbca0e736f888082042f5df051806b9465643e97`；
- 原始压缩包保存在 `data/archives/ACDC/gt_detection_trainval.zip`；
- 安全解压目录为 `data/raw/ACDC_detection`，导入记录为 `IMPORT_RECORD.json`；
- 15 个 COCO JSON 均通过引用、类别、框尺寸、边界和重复 ID 检查；
- 已从官方合并标注生成只包含 fog/rain 的冻结训练和验证子集；
- 自动化测试增至 42 项并全部通过。

## 审计结果

审计产物：

- `runs/data_audit/acdc_detection_audit.json`：机器可读明细；
- `runs/data_audit/acdc_detection_audit.md`：简要表格；
- `scripts/audit_coco_annotations.py`：可复用审计入口。

官方标注的合并训练集包含 1,600 张图像、10,135 个目标；合并验证集包含 406 张图像、2,794 个目标。类别为 person、rider、car、truck、bus、train、motorcycle 和 bicycle。

## 已发现的数据风险

压缩包内的 `gt_detection/rain/instancesonly_rain_train_gt_detection.json` 与合并训练集 `gt_detection/instancesonly_train_gt_detection.json` 的 SHA-256 完全相同。该文件实际包含 fog、night、rain、snow 四种天气的 1,600 张图像，并不是仅含 rain 的 400 张训练图。

因此后续不得直接把这个 rain 训练 JSON 当作雨天子集。雨天训练子集应从合并训练 JSON 按 `file_name` 的首级目录 `rain/` 过滤生成，并保留生成脚本与哈希记录。验证集的 `instancesonly_rain_val_gt_detection.json` 审计正常，可直接使用。

## 冻结的 fog/rain 子集

使用 `scripts/build_acdc_subset.py` 从官方合并 JSON 生成：

- `data/derived/ACDC_detection/fog_rain_train.json`：800 张图、4,329 个框，fog/rain 各 400 张；
- `data/derived/ACDC_detection/fog_rain_val.json`：200 张图、1,403 个框，fog/rain 各 100 张；
- 两个子集结构错误均为 0，且不存在重复 JSON 载荷；
- 审计记录保存在 `runs/data_audit/acdc_fog_rain_subset_audit.json`。

可重复生成命令：

```powershell
python scripts\build_acdc_subset.py --input data\raw\ACDC_detection\gt_detection\instancesonly_train_gt_detection.json --output data\derived\ACDC_detection\fog_rain_train.json --conditions fog rain
python scripts\build_acdc_subset.py --input data\raw\ACDC_detection\gt_detection\instancesonly_val_gt_detection.json --output data\derived\ACDC_detection\fog_rain_val.json --conditions fog rain
```

## 图像包到位后的固定步骤

1. 将 `rgb_anon_trainvaltest.zip` 保存到 `data/archives/ACDC/`；
2. 使用安全导入器解压到 `data/raw/ACDC_images`；
3. 以 `data/raw/ACDC_images/rgb_anon` 为图像根目录，带 `--require-images` 重跑审计；
4. 首轮正式评价只冻结 fog/rain 的 val 集，不使用无公开真值的 test 集形成 AP 结论；
5. 固定 YOLO 配置后依次评价 weather input、DCP、direct、raw conservative、calibrated fusion；
6. 保存逐图预测、每类 AP50、mAP50、漏检率、虚警率和 bootstrap 95% 置信区间。

五方法流水线已经封装为 `scripts/run_acdc_benchmark.py`。脚本会先验证200张 fog/rain 验证图是否全部存在，再递归生成 DCP、直接复原、原始保守融合和校准融合结果，最后统一调用冻结 YOLO 协议；任何阶段失败都会把已执行步骤、返回码和耗时写回 `RUN_MANIFEST.json`，支持根据已有输出续跑。

检测评价现在保存逐图真值、预测框、TP、FP、FN、precision、recall 和 F1。`scripts/summarize_acdc_benchmark.py` 使用相同的图像重采样索引计算各方法的数据集级 mAP50 bootstrap 95% 置信区间及相对 weather input 的配对差值区间，同时生成 fog/rain 分天气结果和每类别明细。汇总前会强制核对图像顺序、标注哈希、检测配置哈希、IoU 阈值和置信度阈值，并校验汇总 mAP 与逐图重算值一致。输出包括 `summary.json`、`summary.csv`、`by_weather.csv`、`per_class.csv` 和 `REPORT.md`。五方法端到端冒烟链路已在嵌套目录样例上实际运行通过；该样例仅证明程序连通性，不作为科研结果。

对应命令：

```powershell
python scripts\import_public_dataset.py --dataset acdc --archive data\archives\ACDC\rgb_anon_trainvaltest.zip --destination data\raw\ACDC_images
python scripts\audit_coco_annotations.py --annotations-root data\raw\ACDC_detection\gt_detection --images-dir data\raw\ACDC_images\rgb_anon --require-images --output runs\data_audit\acdc_with_images_audit.json --markdown-output runs\data_audit\acdc_with_images_audit.md
```
