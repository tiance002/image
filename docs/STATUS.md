# 当前状态（聚合公开源码副本）

2026-10-25：保留v4@48k基线，C0/C1各4000步已完成，当前GT配方不解决bus/train，不延长、不替换默认权重、不push。

新标准conf.001 AP50主200：input.256338、v4.251481、guard.251884、C0.249156、C1.252121；全406：.233510/.220924/.224067/.214648/.215576。不得与旧.25截断AP混比。

主bus C0.148824→C1.148829，CI跨0；主train.283577→.284203，CI跨0。全部95个bus/train GT在runtime.25下C0/C1命中集合完全相同；全406 C1相对input的bus/train区间为负，train下降7.132pp。

已修复并验证颜色、零annotation ID适配、标准AP/runtime协议和NAF非整除残差。训练前后冻结YOLO及BN不变，完整输入/GT流一致；27相关测试通过。所有模型、数据、原始预测保留在科研工作区。

详见[最终决策](../results/next_phase/FINAL_DECISION.md)、[P0](../results/next_phase/P0_REPORT.md)、[完整结果](../results/next_phase/TRAINING_REPORT.md)、[复现与范围](../results/next_phase/SNAPSHOT_SCOPE.md)。外部两个官方基线及LPIPS因资源缺失BLOCKED；本轮不是独立test、多seed或完整D1验收。
