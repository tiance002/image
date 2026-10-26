# 复核本阶段交付（v2）

运行区：`<local research workspace>`，Python：`<local yolo_daily Python>`。Git版本核对区是`runs/github_publication/staging`，不是数据和权重运行区。所有新产物都位于本目录。

## 推荐：只读复核

在真实科研区执行（PowerShell）：

```powershell
$env:PYTHONPATH='third_party/pylibs;src;.'
$env:PYTHONIOENCODING='utf-8'
& '<local yolo_daily Python>' 'results/np2_revision/bus_train_diagnosis_20261026/validate_delivery.py'
```

该脚本只用标准库：核对37项输入/权重/源码/预测哈希、64张原图哈希、192批/768组记录的向量误差、原生损失分解、每个正anchor索引与640网格坐标、5376行逐项梯度表、95行目标矩阵。不会导入Torch、运行模型或重新计算梯度向量。原始向量一致性由已保存的实际autograd结果与执行源码共同提供；不把离线查表声称为重新反传。

## 本轮实际运行的程序

- `offline_audit.py`：读取已有JSON、GT及正式预测，重算官方COCO匹配并输出覆盖、矩阵与案例链接；不运行模型。
- `component_gradient_probe_fp64_metrics.py`：最终有效FP64批次的最初11个v4批，逐项反传，并额外计算独立类总梯度。
- `component_gradient_probe_fp64_minimal.py`：保留这11个有效批次SHA后继续v4，并完成C0/C1。每类四项均实际autograd；空正集合的导数严格为0。移除已被28项/原生总量及3个原生分量检查覆盖的冗余类总反向，保留所有核心向量校验。最终目录是`fp64_component_gradients/`。
- `assemble_report_v2.py`：最终有效64×3完成后生成报告，检查输入/参数不变；相同已存在产物仅做内容一致性检查，拒绝覆盖不同内容。
- `contract_checks_source.py`：保留本轮已执行的5项mask、4项FP64归约检查逻辑。记录见两份checks JSON；此源文件只额外做过编译检查，没有重复运行已通过套件。

组件求导源文件无优化器、无训练循环、无NMS预测生成；全部数据来自原64张train图片，固定顺序/无随机翻转。当前模型前向保留历史TF32设置，反向卷积TF32关闭；FP64只用于CPU统计归约，模型与梯度仍FP32。

分类正/负按target_scores是否大于0分组，不是梯度符号。分类正项不保证提高分类置信度。表中原生任务梯度没有乘历史C1的外部GT系数，也没有合并复原A流梯度。

## 输出保护与旧数值版本

程序默认拒绝原地重复生成已有实验产物。需要独立重复自动求导时，应先由负责人审查范围，使用本目录同层的新独占目录放置执行源和同一preflight/manifest，不改数据、检查点或已有结果；同层路径深度用于源码ROOT解析。不同硬件不要求向量SHA逐位相等，校验阈值和原生协议保持固定。不要在公开Git副本中运行来替代科研区。

`component_batches_v4/`下21个初期完成记录未进入最终统计：CPU FP32点积/范数归约曾产生无效余弦>1。所有源、记录、资源续跑日志均保留，`PARTIAL_GRADIENT_METRIC_STATUS.json`规定其使用边界。最终FP64有效批与这些记录完全分开。资源中断不作为多seed或独立实验。

`supervision_probe.json`等上一轮NP2-R输入仍保留；历史正确的48个batch4标量和192个类总梯度向量检查仅离线复核。本阶段不重复原7项监督/12项恒等融合、颜色/数据生成、Histoformer或其他复原推理。

## 证据位置

- 最终结论：`BUS_TRAIN_DIAGNOSIS_REPORT_v2.md`。
- 每项原生GT→anchor：`native_positive_anchor_trace.csv`与各批JSON。
- 逐图/类/项loss与参数梯度：`component_gradients_per_image.csv`。
- 每批向量比较：`component_vector_checks.csv`。
- 95个bus/train目标四方法并排：`bus_train_gt_matrix_all406.csv`。
- 完整候选/定位/类别观察：`validation_cases.json`；NMS均保留未解决状态，未保存原生NMS索引。
- 同一训练GT链接：`same_train_gt_evidence_links.json`；跨split链接只描述序列关系，不能用数值ID硬连。
- 版本和输入：`preflight.json`、`sample_image_hashes.json`、`preservation_checks.json`。
- 唯一候选实验：`next_minimal_experiment_v2.json`，状态PROPOSED_ONLY_NOT_EXECUTED；当前总体根因状态BLOCKED_ROOT_CAUSE_UNRESOLVED。

本阶段不提交/推送、merge、部署或替换默认权重。报告完成即停止等待审查。

## 原生候选几何补充

初稿与错误附加断言保留，状态见REPORT_STATUS.md。最终validate_delivery.py按照native_candidate_geometry_audit.json中由实际固定原生方法重放的候选mask核对，而非要求正anchor中心在原始GT框内。tal.py/loss.py新增源码SHA位于preflight_native_source_addendum.json，离线校验同样检查这些SHA。几何方法没有模型前向、没有重新做有预测输入的TAL分配，更没有更新权重；原64梯度结果不重跑。

两个零正bus均有5个原生几何候选，具体top-k/冲突/质量打分原因仍未保存；最终唯一下一步是固定C1及原64张的原生阶段只读追踪。原tile候选仅存档，不自动执行。

原生几何重放的可复核源码保存为`native_geometry_replay.py`，仅调用固定原生select_candidates_in_gts，不调用模型或分配器forward；本轮几何审计已执行，保存此独立源码后只做编译检查，未重复重放。
