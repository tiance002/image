# 原生候选几何勘误

本地固定Ultralytics版本：8.4.104。
原生实现：`<local Ultralytics package>/utils/tal.py`。
SHA256：`6294831f4aea3eb883700cd30d34ce2db5369e31167a42a6d20b3e716c984a2c`。
完整函数及criterion构造器摘录：PINNED_NATIVE_TAL_SOURCE.txt。

select_candidates_in_gts对宽/高分别判断：小于stride[0]=8时，只在候选中心筛选框中替换为stride_val=stride[1]=16；回归target_bboxes保持原GT。这是实际固定版本的行为，不能套用其他版本“原框内中心”的概括。

本阶段最终校验曾错误要求所有正中心都在原框内，失败后保留原断言与初始记录；没有改动原生分配器或loss。使用同一原生几何方法、完整图内全部GT及既有640像素框离线重放，7415个实际正anchor全部满足原生候选规则。650个不满足原框严格内部（含边界）；其中607个超出1e-4容差。这不是7415次新推理，而是对192批真实分配中选出的7415个索引做离线几何核对。

train image766的两个bus：

| source/eval GT | 原框中心数8/16/32 | 原生候选数8/16/32 | 最终正anchor |
|---|---|---|---|
|5123/5124|1/0/0|4/1/0|0|
|5124/5125|0/0/0|4/1/0|0|

两个都有5个原生几何候选，不能把零正归因为无候选。最终占用分别显示5/5和3/5候选被其他GT分配，第二个另2个未分配；这只是结束状态，不证明bus经过top-k后被冲突步骤抢走。必须有原生align metric、top-k、冲突前mask_pos与最终消解trace才能区分。当前总体及局部阶段根因仍UNRESOLVED。

旧candidate_centers_in_ignore统计的是原框内部点与BCE-ignore掩码的交集，不是原生扩展候选的忽略计数。BCE-ignore不输入TAL，且target_scores>0的正项会被保留；不能把这些字段误读为零分配由忽略掩码造成。

原始损失、梯度向量、GT与预测表不受这次解释勘误影响，不重算模型前向。最终结论使用v2报告与v2检查记录。
