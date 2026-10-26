# 本阶段产物状态：以v2为准

最终审查入口：**BUS_TRAIN_DIAGNOSIS_REPORT_v2.md**。
最终验收：**final_checks_v2.json + review_validation_v2.json**。
最终唯一下一步：**next_minimal_experiment_v2.json**（固定C1/原64张的原生TAL阶段只读追踪；未执行）。
复核说明：**REPRODUCE_v2.md**。

## 保留但不作为最终结论的初稿

- BUS_TRAIN_DIAGNOSIS_REPORT.md：旧版本待修订。其损失/梯度/正式预测数值仍有效；“原框没有中心可导致原生候选缺口”的局部解释不正确。v2已撤回，并补原生小框候选扩展规则。
- final_checks.json：仅是初稿装配检查，不是最终离线几何验收；由v2两份checks覆盖最终判断。
- next_minimal_experiment.json：tile640/1024历史候选保留，当前优先级被最终TAL阶段追踪建议覆盖。本阶段未执行任一候选。
- REPRODUCE.md、assemble_report.py：初稿复现说明/生成器，保留原文；请使用v2入口。
- native_center_containment_investigation.json：发现607个明显原框外正中心时的初始UNRESOLVED记录；已由native_candidate_geometry_audit.json查明原生候选扩展，不代表原生损失/梯度失败。
- validate_delivery_original_gt_containment.py.txt：保留最初错误的附加断言。当前validate_delivery.py按实际固定版本的原生候选mask核对，未放宽原生损失或梯度阈值。
- component_batches_v4/：21个部分完成的初期诊断批，CPU FP32余弦归约有缺陷，旧版本待修订。最终统计只读取fp64_component_gradients/。
- assemble_report_v2.syntax_draft.py.txt：修订报告生成源码首次写入时的字符串换行语法错误草稿，未运行或产生实验结果；已修复并编译后生成v2。

原生梯度数据、共享CSV/JSON及旧NP2-R目录均保留。没有模型/分配器/训练loss改动，没有训练、下载、推送或部署。
