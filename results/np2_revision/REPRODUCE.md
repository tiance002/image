# NP2-R 复现入口与保留规则

运行区必须含原始资源；源码Git副本没有数据与权重。Python固定D:/Drivers/anaconda/envs/yolo_daily/python.exe，工作目录D:/数字图像处理/SafeRestore-RF。PowerShell设置`$env:PYTHONPATH='third_party/pylibs;src;.'`与`$env:PYTHONIOENCODING='utf-8'`；GitBash的等价设置是`export PYTHONPATH="third_party/pylibs;src;."`。本次工具的shell选择器意外启动WSL；长任务采用PowerShell等价环境，之后显式调用C:/Program Files/Git/bin/bash.exe可正常运行Windows Git Bash，已验证真实cwd和PYTHONPATH一致。工具已关闭沙箱，没有dangerouslyDisableSandbox字段，写入后均核对落盘。

当前canonical结果已用于本轮执行，不能原地重复下列命令覆盖；完整复现需在准备好同样本地资源的独立项目副本中，使用空的新输出空间。各脚本的--out/--out-root可指定新目录；固定汇总脚本则需在独立副本的空results/np2_revision中执行。

1. `scripts/np2_histoformer_restore.py --weights third_party/weights/histoformer_official_net_g_real.pth --name histoformer_real_fusionfix --weather rain snow --out-root results/np2_revision/restored`：运行前断言目标目录没有PNG，使用同官方权重重新前向，不放大旧量化PNG。
2. `scripts/np2_detect.py --name histoformer_real_fusionfix --images results/np2_revision/restored/histoformer_real_fusionfix --out-root results/np2_revision/predictions`：必须显式指定新输出根目录，先验证200张与metadata200，再运行冻结YOLO。
3. `scripts/np2r_supervision_probe.py --grad --batch 4 --grad-batch 1`：原固定64/431GT，v4/C0/C1逐批loss与完整64梯度批向量校验；无优化器。
4. `scripts/np2_scale_order_probe.py`：原固定24，主D/A/B统一FP32与几何，额外量化/灰边/bf16/实际rect路径独立记录；读取已有检查点，绝无训练。
5. `scripts/np2r_quality.py --mode both`：固定1024裁剪复核与受影响dev200重新推理，内部input/v4/C0/C1质量直接复用；本地LPIPS缓存，不下载。
6. `scripts/np2r_summarize.py --resamples 1000`：CPU复算同子集AP/runtime/双协议GT矩阵/修复前后PNG统计及区间，复用input/v4/C0/C1 val预测。
7. 审阅真实数字后建立`next_experiment.json`，只建议一项且not_executed=true；`scripts/np2r_revision_report.py --validate-only`校验后生成六节报告，再用`scripts/np2r_finalize.py`核验历史417文件与5项权重未变。

本轮不重复交接中已通过的12恒等用例和7监督数学用例；`scripts/np2r_scale_checks.py`为新增尺度CPU检查入口（包含实际rect路由时加--production-rect），证据在scale_probe_cpu_checks.json。完整GPU诊断本身逐批核验，不用汇总巧合代替。

AP候选协议conf=.001/maxDets100与运行时conf=.25/maxDets300分别保存；GT评估ID=源annotation id+1。任何hit/miss必须注明协议。样本来自历史val/train及相关序列，不构成独立test。

补充数值修订：首次完整监督诊断的标量校验通过、梯度向量校验失败，结果保留为supervision_probe.failed_tf32_v1.json及supervision_run.log。tf32_gradient_diagnosis.py在同一固定首图证明cuDNN TF32反向导致分解反传的有限精度非可加性。最终监督脚本保留历史前向/分配，只在反向关闭卷积TF32，并保留enabled/benchmark/deterministic；1e-4阈值不变。完整合格运行日志是supervision_run_precisionfix.log，不得引用失败smoke充当64张结果。主尺度对照仍是前后向统一FP32且TF32关闭的独立配置。
