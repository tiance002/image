# 旧版本待修订：NP2-R结果有效性登记

审查基线为next-phase-gt-20261025 / 3d53ca0dbd9902c86de6953de791a73c557dd2ef。原results/np2_external_probe全部文件保留，未原地改写；本登记优先于其中受影响的推断。

| 原始证据 | 当前状态 | 替代证据 |
|---|---|---|
| histoformer_real雨/雪200预测、指标、区间、GT变化及dev画质 | 旧版本待修订；分块融合归一化错误，禁止用作有效模型成绩 | 本目录fusionfix重新推理、冻结检测、metrics/bootstrap/quality文件 |
| tiling_consistency.json局部最大差及上下文解释 | 旧版本待修订；融合错误混入，不能直接归因于上下文 | tiling_consistency_fusionfix.json |
| supervision_probe.json逐类box/DFL与grad_norm | 旧版本待修订；网格/像素混用及[N]广播；旧梯度范数不可加和 | supervision_probe.json修订版逐批loss/向量校验 |
| supervision_probe.json原生总损失、同一原生联合分配的正anchor覆盖 | 可保留；本轮逐GT核对，不代表整个训练期充分 | 修订版保留原64张、同检查点 |
| scale_order_probe.json旧A/B结果 | 仅历史混杂管线：精度、量化、灰边、上下文均未隔离 | 修订版固定24、A/B/D FP32共同几何及独立扰动 |
| input/v4/C0/C1正式val预测及原有内部画质结果 | 不受本轮两处统计/融合缺陷影响，复用不重推理 | 来源文件SHA见preflight、metrics与quality |
| bus_train_gt_matrix.csv原有input/v4/C0/C1运行时hit/miss | conf=.25口径仍有效；旧Histoformer列待修订 | 本目录按image_id+GT_id+协议去重矩阵 |

GT2730：旧v4在AP候选协议conf=.001/maxDets100命中，在运行时conf=.25/maxDets300未命中；协议不同，不能交叉引用。其源标注ID为2729，正值评估GT ID为2730。

撤回“监督已充分/已排除覆盖问题”“证明主要是小目标被抹掉”“单个外部权重近似恒等可排除更强预训练路线”“区间跨零证明不更差”“裁剪C2失败证明尺度顺序不是主因”。只保留有对应样本与协议的观察。诊断样本中有监督不等于整个训练期充分；梯度范数比例名称为“梯度范数占比（不可加和）”。

本轮不执行C2、不启动任何训练，最终结论等待本目录完整修订报告。
