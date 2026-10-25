# 成熟模型复用与对照资源核查

核查日期：2026-10-25。本文记录官方文本、源码和本机资源核查；外部模型没有执行推理。新协议下本地模型的实测指标由同目录 `P0_REPORT.md` 和主报告提供，不能用下表的空缺推导性能结论。

## 结论与执行状态

| 方法 | 角色 | 本轮状态 | 可报告的本轮外部模型实测指标 |
|---|---|---|---|
| input | 不复原的检测参照 | 由统一评估流程实测 | 见 P0 报告 |
| NAFNet v4@48k | 保留基线 | 使用本地历史权重 | 见 P0 报告 |
| guard@52k | 上轮置信度约束候选 | 使用本地历史权重，不替换 v4 | 见 P0 报告 |
| C0 / C1 | 本轮无/有真实 GT 检测监督的公平续训 | 训练就绪与实测步数以 `TRAINING_REPORT.md` 为准 | 本文不填造数值 |
| AdaIR-5D | 成熟五退化复原对照 | **BLOCKED：未找到官方本地权重，且缺依赖** | AP、PSNR、SSIM、LPIPS、延迟、参数量、显存均未实测 |
| Histoformer-Allweather | 成熟雨雪复原对照 | **BLOCKED：未找到官方本地权重，且缺依赖** | 同上，不能以论文其他数据集结果代替 |

本轮遵守原有节流约束：只访问少量官方网页、代码文本和文件元数据；未下载数据集、权重、代码压缩包或依赖，没有启动随机外部模型、从零训练或调用云端模型替代本地对照。缺资源是 P3 实测的真实阻断，不妨碍其余本地工作继续。

## 官方机制核查

UniRestore 的官方 `DetectionLoss` 冻结 torchvision 的 RetinaNet/Faster R-CNN，向检测器传入复原图和真实 `boxes`/`labels`，求原生检测损失之和；其学生前向保留图像梯度。这支持采用真实框监督复原的方向，但不是 YOLOv8 或 DFL 实现。本项目的 cls/box/DFL 与任务分配应复用本机 Ultralytics 原生实现。官方模块会调用 `.train()`，本轮还需单独固定检测器 BatchNorm，不能直接复制这段状态管理。[官方 DetectionLoss](https://github.com/unirestore/UniRestore/blob/6457cd1406e03be627dd06710406169671d89afb/src/core/base/eval_detection.py)

UniRestore 当前提供的 stage2 默认任务为 `ir/cls/seg`；扩展检测任务需要任务目标和数据接口。此处仅借鉴其任务损失思想，没有迁移扩散骨干或宣称复现整套 UniRestore。代码许可证为 MIT。[stage2 配置](https://github.com/unirestore/UniRestore/blob/6457cd1406e03be627dd06710406169671d89afb/configs/train_stage2.yaml)、[许可证](https://github.com/unirestore/UniRestore/blob/6457cd1406e03be627dd06710406169671d89afb/LICENSE)

## AdaIR-5D

官方项目为 `c-yn/AdaIR`，核查提交 `ccb8b98e49614e07badd0641e5163fa7635c2f02`。五退化权重面向去噪、去雨、去雾、去模糊及低照度增强；**不含去雪**。官方测试默认检查点为 `ckpt/adair5d.ckpt`，`--mode 6` 表示五任务评测；多个 mode 不代表可按验证 GT 逐图选择不同网络。[官方 README](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/README.md)、[测试入口](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/test.py)

官方权重入口为 [Google Drive 文件夹](https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH?usp=sharing)。核查文本未给出大小，本轮未估算或下载。仓库代码采用 MIT；远端权重文件尚未取回，不能声称校验过权重哈希或额外条款。[许可证](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/LICENSE)

预处理读取 RGB，`ToTensor` 转为 `[0,1]`，测试数据中心裁到 16 的倍数；不是 BGR，也没有 ImageNet 均值方差归一化。若后续按官方裁剪运行检测评估，必须把输出放回原图坐标：被裁掉的边缘保留输入，或单独声明有效视野，禁止静默改变 GT。若改为 pad/unpad，则需标为适配规则并先核查等价性。[数据预处理](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/utils/dataset_utils.py)、[crop_img](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/utils/image_utils.py)

预先固定的适用范围为 fog/rain/night 使用同一个 `adair5d.ckpt`，snow 标为不支持。若需要 406 张总表，可另建明确命名的“支持天气用 AdaIR，其余保持 input”组合；其数值不能冒充纯 AdaIR 全天气模型结果，也不得按 GT 决定回退。

## Histoformer-Allweather

官方项目为 `sunshangquan/Histoformer`，核查提交 `1f045f06c03551c31504d8042dbe6ff9b9569108`。Allweather 配置覆盖 RainDrop、Outdoor-Rain、Snow100K；官方 benchmark 命令使用 `net_g_best.pth`，真实雪示例使用 `net_g_real.pth`。本轮不将其视为夜间增强模型；也没有从已读配置中确认独立雾天权重，不能因名称 Allweather 就宣称所有天气均适用。[官方 README](https://github.com/sunshangquan/Histoformer/blob/1f045f06c03551c31504d8042dbe6ff9b9569108/README.md)、[模型配置](https://github.com/sunshangquan/Histoformer/blob/1f045f06c03551c31504d8042dbe6ff9b9569108/Allweather/Options/Allweather_Histoformer.yml)

官方 [Drive 权重入口](https://drive.google.com/drive/folders/1dmPhr8Z5iPRx9lh7TwdUFPSfwGIxp5l0?usp=drive_link) 和仓库链接的 [Hugging Face 模型页](https://huggingface.co/sunsean/Histoformer) 已核验。HF 元数据提交 `f6acde20acaa26ef5e235f2c71ff7637f4d9a945` 声明 `license: mit`，提供 `Allweather/pretrained_models/net_g_real.pth`，文件大小 **66,603,665 字节（约 63.52 MiB）**；当前元数据没有 `net_g_best.pth`。GitHub 递归目录没有独立 LICENSE 文件，因此仅记录官方 HF 的许可声明，不把第三方 BasicSR 等依赖一并视为已授权重新发布。未下载该文件。[官方模型元数据 API](https://huggingface.co/api/models/sunsean/Histoformer?blobs=true)

官方推理把 OpenCV BGR 转 RGB，除以 255，完整视野反射填充到 8 的倍数；前向后取消填充，截断 `[0,1]`，转回 BGR 保存。加载的是检查点 `params`，网络结构来自上述 YAML；后续适配应保留这些规则。[推理脚本](https://github.com/sunshangquan/Histoformer/blob/1f045f06c03551c31504d8042dbe6ff9b9569108/Allweather/test_histoformer.py)、[图像 I/O](https://github.com/sunshangquan/Histoformer/blob/1f045f06c03551c31504d8042dbe6ff9b9569108/Allweather/util.py)

预先固定的 Allweather 基准选择为 rain/snow 均用 `net_g_best.pth`；`net_g_real.pth` 可作为另一个提前声明的真实雪专项模型，不能看验证分数后换权重。fog/night 暂列为未确认适用/不支持。两者均没有本地权重，当前没有产生任何输出图像或性能数据。

## 本机资源与兼容性证据

遍历范围限定在 `D:/数字图像处理`、用户 `.cache/huggingface`、`.cache/torch`；排除 `.git` 和 `__pycache__`。工作区发现 58 个 `.pt/.pth/.ckpt/.safetensors` 文件，路径属于 YOLO、NAFNet、旧工程模型及测试夹具，没有 AdaIR/Histoformer 命名路径；数据目录也没有上述目标检查点。HF hub 仅发现 faster-whisper-base 缓存；torch 缓存目录不存在。该结论限定于已查根目录，不能排除未告知位置的其他磁盘文件。

本轮本地解释器为 Python 3.11.15、PyTorch 2.6.0+cu124、Ultralytics 8.4.104、NumPy 2.4.4。模块探测显示 `einops`、`timm`、`lightning`、`pytorch_lightning`、`torchmetrics`、`basicsr`、`lpips`、`yacs`、`natsort` 均不可导入。AdaIR 官方架构依赖 einops，官方测试包装依赖 lightning；Histoformer 官方入口依赖仓库内 BasicSR、einops、natsort。因此现环境不能直接运行其官方入口。**这证明缺依赖，不证明新版本 PyTorch 必然不兼容**；没有通过降级或安装修改当前训练环境。[AdaIR 架构](https://github.com/c-yn/AdaIR/blob/ccb8b98e49614e07badd0641e5163fa7635c2f02/net/model.py)、[Histoformer 安装说明](https://github.com/sunshangquan/Histoformer/blob/1f045f06c03551c31504d8042dbe6ff9b9569108/INSTALL.md)

外部模型的参数量、端到端延迟、峰值显存和图像质量全部保持 **未实测**。参数个数即便能从官方结构推导，也不能据此冒充已加载官方权重和已完成推理；论文其他平台的效率也不能用于本机横向排名。待资源到位，应固定输出格式，再共用同一 YOLOv8n、640 输入、NMS、AP 阈值 0.001 与运行时阈值 0.25；同时测复原+检测全链路并记录 warmup/同步/峰值显存方法。

## 追溯与下一步

官方链接、核查提交、文件 blob SHA、权重元数据、本机模块清单及固定天气选择规则保存在 `official_sources.json`。远端 `tiance002/image/main` 通过只读 GitHub 接口核验，仍为 `396af4a50a2512b7dabb00a69b3ccfded403dea9`；没有推送或修改远端仓库。

P3 解除阻断需要事先备好的官方权重及运行依赖。当前不能判断外部模型是否同时超过 v4 的图像质量和检测指标，也没有依据建议立即更换 NAFNet 骨干。继续完成本地正确性核查、真实 GT 损失与公平对照后，再决定是否值得为外部基线分配下载流量。

<!-- LOCAL_MEASURED_RESULTS -->
# 本轮本地模型实测对照

外部两基线仍为BLOCKED，不填写或推断其数值。以下统一新协议本地数值可追溯metrics.json；完整逐天气/训练/区间见TRAINING_REPORT.md。

## rain_fog200

| 方法 | AP50(.001) | AP50:95(.001) | 去train六类AP50 | Recall(.25) | FP/图(.25) | OPR(.25) |
|---|---:|---:|---:|---:|---:|---:|
| input | 0.256338 | 0.155289 | 0.245156 | 0.433186 | 1.250000 | 1.000000 |
| v4 | 0.251481 | 0.150662 | 0.240511 | 0.445668 | 1.225000 | 0.957627 |
| C0 | 0.249156 | 0.145423 | 0.243420 | 0.439794 | 1.270000 | 0.947458 |
| C1 | 0.252121 | 0.150022 | 0.246774 | 0.446402 | 1.260000 | 0.952542 |
| guard | 0.251884 | 0.151387 | 0.240550 | 0.450073 | 1.245000 | 0.976271 |

| 类别 | 非crowd GT | input AP50 | v4 AP50 | C0 AP50 | C1 AP50 | guard AP50 |
|---|---:|---:|---:|---:|---:|---:|
| person | 194 | 0.290197 | 0.310560 | 0.301416 | 0.307387 | 0.302375 |
| bicycle | 20 | 0.213875 | 0.181844 | 0.188087 | 0.172491 | 0.185772 |
| car | 1019 | 0.611768 | 0.615754 | 0.616231 | 0.621489 | 0.620442 |
| motorcycle | 23 | 0.032247 | 0.059566 | 0.088271 | 0.111268 | 0.061143 |
| bus | 37 | 0.173570 | 0.138274 | 0.148824 | 0.148829 | 0.137173 |
| train | 12 | 0.323432 | 0.317303 | 0.283577 | 0.284203 | 0.319891 |
| truck | 57 | 0.149280 | 0.137066 | 0.117688 | 0.119180 | 0.136394 |

## all406

| 方法 | AP50(.001) | AP50:95(.001) | 去train六类AP50 | Recall(.25) | FP/图(.25) | OPR(.25) |
|---|---:|---:|---:|---:|---:|---:|
| input | 0.233510 | 0.138682 | 0.208627 | 0.376212 | 1.009852 | 1.000000 |
| v4 | 0.220924 | 0.127568 | 0.198921 | 0.381432 | 1.051724 | 0.952428 |
| C0 | 0.214648 | 0.126094 | 0.198148 | 0.376957 | 1.051724 | 0.948464 |
| C1 | 0.215576 | 0.127859 | 0.199590 | 0.383296 | 1.056650 | 0.953419 |
| guard | 0.224067 | 0.129733 | 0.203710 | 0.388143 | 1.061576 | 0.970268 |

| 类别 | 非crowd GT | input AP50 | v4 AP50 | C0 AP50 | C1 AP50 | guard AP50 |
|---|---:|---:|---:|---:|---:|---:|
| person | 518 | 0.260138 | 0.269798 | 0.265835 | 0.269461 | 0.269081 |
| bicycle | 68 | 0.081542 | 0.067000 | 0.061383 | 0.055107 | 0.064568 |
| car | 1836 | 0.557207 | 0.555463 | 0.556141 | 0.563299 | 0.561791 |
| motorcycle | 59 | 0.072163 | 0.063184 | 0.070210 | 0.070472 | 0.081306 |
| bus | 49 | 0.121324 | 0.088846 | 0.097160 | 0.099366 | 0.094568 |
| train | 46 | 0.382810 | 0.352937 | 0.313652 | 0.311490 | 0.346211 |
| truck | 106 | 0.159388 | 0.149238 | 0.138155 | 0.139836 | 0.150943 |

## 有效像素对齐数据的画质

同一清晰源图合成退化的像素对齐dev（fog100/rain100），不是ACDC真实跨时刻weather/reference配对；不能推断真实夜雪画质

下面为每图指标再平均；RGB范围0..1，PSNR全像素，SSIM高斯11/sigma1.5的valid窗口。LPIPS因本地包及预训练权重缺失而BLOCKED。没有把ACDC跨时刻weather/reference直接计算成配对画质。

| 方法 | dev200 PSNR(dB) | dev200 SSIM | fog PSNR / SSIM | rain PSNR / SSIM |
|---|---:|---:|---|---|
| input | 18.265963 | 0.809295 | 13.4557 / 0.764636 | 23.0762 / 0.853954 |
| v4 | 27.766425 | 0.939719 | 27.7045 / 0.943455 | 27.8283 / 0.935983 |
| C0 | 28.495759 | 0.943494 | 27.9708 / 0.944984 | 29.0207 / 0.942005 |
| C1 | 28.430017 | 0.943198 | 27.9530 / 0.944853 | 28.9070 / 0.941542 |
| guard | 27.732876 | 0.939343 | 27.4872 / 0.941948 | 27.9785 / 0.936739 |

| 画质差值(dev200) | 均值差 | 配对bootstrap95%区间 |
|---|---:|---|
| C1:C0 psnr | -0.065742 | [-0.088619, -0.044627] |
| C1:C0 ssim | -0.000297 | [-0.000378, -0.000225] |
| C1:v4 psnr | 0.663592 | [0.451043, 0.879216] |
| C1:v4 ssim | 0.003479 | [0.002597, 0.004462] |
| C0:v4 psnr | 0.729333 | [0.516164, 0.947006] |
| C0:v4 ssim | 0.003776 | [0.002882, 0.004765] |

画质区间基于对齐合成dev图像的配对重采样；不替代真实天气主观质量、LPIPS、多seed或场景独立评测。

## 完整端到端实测

Each weather first10 paths lexicographically, GT-independent; five warmup images, then40 timed images; warm filesystem cache

file read/decode, RGB conversion, H2D, bf16 NAFNet tiled restoration, quantize/D2H, BGR conversion, FP32 YOLO640 .25 NMS .7 maxDet300, CPU boxes transfer

| 方法 | 复原参数 / 检测参数 | 平均ms | 中位ms | P95 ms | 峰值allocated MiB |
|---|---:|---:|---:|---:|---:|
| input | 0 / 3157200 | 514.04 | 452.17 | 1160.12 | 38.4 |
| v4 | 17061219 / 3157200 | 1772.82 | 1453.45 | 3306.33 | 1158.5 |
| C0 | 17061219 / 3157200 | 1685.31 | 1416.32 | 3401.32 | 1145.1 |
| C1 | 17061219 / 3157200 | 1370.58 | 1272.61 | 1871.67 | 1157.4 |
| guard | 17061219 / 3157200 | 2112.48 | 1905.49 | 3451.58 | 1169.9 |

模型载入、权重哈希、写图和GUI不计时。全部本地NAFNet候选结构相同，延迟差异不应解释成骨干加速；CPU bootstrap已结束后才计时。最初全406评估metadata另有不含读图的计时，两者边界不同，不能混用。

## 检测配对区间：bootstrap_p0.json

paired image, stratified by weather, percentile95

相邻帧相关；图像重采样不构成场景独立或多随机种子证据。区间跨零表示证据不足，不证明无损。无真值类别重采样记为NaN并报告有效次数。

| 集合 | 差值(candidate:reference) | 指标 | 点差 | 95%区间 | 有效重采样 |
|---|---|---|---:|---|---:|
| rain_fog200 | v4:input | map50 | -0.004857 | [-0.019976, 0.011749] | 1000 |
| rain_fog200 | v4:input | map50_95 | -0.004627 | [-0.014701, 0.004178] | 1000 |
| rain_fog200 | v4:input | bus | -0.035296 | [-0.098519, 0.006236] | 1000 |
| rain_fog200 | v4:input | train | -0.006129 | [-0.026406, 0.000000] | 1000 |
| rain_fog200 | v4:input | truck | -0.012214 | [-0.038638, 0.006947] | 1000 |
| rain_fog200 | v4:input | recall | 0.012482 | [0.000682, 0.025022] | 1000 |
| rain_fog200 | v4:input | fp_per_image | -0.025000 | [-0.140000, 0.080000] | 1000 |
| rain_fog200 | v4:input | opr | -0.042373 | [-0.058718, -0.027415] | 1000 |
| rain_fog200 | guard:input | map50 | -0.004454 | [-0.018234, 0.011956] | 1000 |
| rain_fog200 | guard:input | map50_95 | -0.003902 | [-0.013764, 0.005092] | 1000 |
| rain_fog200 | guard:input | bus | -0.036397 | [-0.096978, 0.000110] | 1000 |
| rain_fog200 | guard:input | train | -0.003542 | [-0.017410, 0.000000] | 1000 |
| rain_fog200 | guard:input | truck | -0.012886 | [-0.035880, 0.003195] | 1000 |
| rain_fog200 | guard:input | recall | 0.016887 | [0.006367, 0.028975] | 1000 |
| rain_fog200 | guard:input | fp_per_image | -0.005000 | [-0.110000, 0.110000] | 1000 |
| rain_fog200 | guard:input | opr | -0.023729 | [-0.036083, -0.013626] | 1000 |
| all406 | v4:input | map50 | -0.012587 | [-0.023354, -0.002684] | 1000 |
| all406 | v4:input | map50_95 | -0.011114 | [-0.019353, -0.004464] | 1000 |
| all406 | v4:input | bus | -0.032478 | [-0.079448, -0.004303] | 1000 |
| all406 | v4:input | train | -0.029873 | [-0.076895, 0.006981] | 1000 |
| all406 | v4:input | truck | -0.010150 | [-0.031652, 0.007294] | 1000 |
| all406 | v4:input | recall | 0.005220 | [-0.002542, 0.013110] | 1000 |
| all406 | v4:input | fp_per_image | 0.041872 | [-0.034483, 0.118227] | 1000 |
| all406 | v4:input | opr | -0.047572 | [-0.061856, -0.035265] | 1000 |
| all406 | guard:input | map50 | -0.009443 | [-0.019529, -0.000190] | 1000 |
| all406 | guard:input | map50_95 | -0.008949 | [-0.017556, -0.002816] | 1000 |
| all406 | guard:input | bus | -0.026756 | [-0.075744, 0.003580] | 1000 |
| all406 | guard:input | train | -0.036599 | [-0.084361, 0.007052] | 1000 |
| all406 | guard:input | truck | -0.008444 | [-0.028133, 0.007520] | 1000 |
| all406 | guard:input | recall | 0.011931 | [0.005185, 0.019313] | 1000 |
| all406 | guard:input | fp_per_image | 0.051724 | [-0.014778, 0.120751] | 1000 |
| all406 | guard:input | opr | -0.029732 | [-0.040283, -0.019749] | 1000 |

## 检测配对区间：bootstrap_gt.json

paired image, stratified by weather, percentile95

相邻帧相关；图像重采样不构成场景独立或多随机种子证据。区间跨零表示证据不足，不证明无损。无真值类别重采样记为NaN并报告有效次数。

| 集合 | 差值(candidate:reference) | 指标 | 点差 | 95%区间 | 有效重采样 |
|---|---|---|---:|---|---:|
| rain_fog200 | C1:C0 | map50 | 0.002965 | [-0.004840, 0.011534] | 1000 |
| rain_fog200 | C1:C0 | map50_95 | 0.004599 | [0.000967, 0.009287] | 1000 |
| rain_fog200 | C1:C0 | bus | 0.000005 | [-0.012055, 0.004494] | 1000 |
| rain_fog200 | C1:C0 | train | 0.000626 | [-0.004369, 0.005847] | 1000 |
| rain_fog200 | C1:C0 | truck | 0.001492 | [-0.009184, 0.013573] | 1000 |
| rain_fog200 | C1:C0 | recall | 0.006608 | [0.000726, 0.012831] | 1000 |
| rain_fog200 | C1:C0 | fp_per_image | -0.010000 | [-0.075000, 0.050125] | 1000 |
| rain_fog200 | C1:C0 | opr | 0.005085 | [-0.005094, 0.014977] | 1000 |
| rain_fog200 | C1:input | map50 | -0.004217 | [-0.030423, 0.017957] | 1000 |
| rain_fog200 | C1:input | map50_95 | -0.005267 | [-0.026174, 0.006662] | 1000 |
| rain_fog200 | C1:input | bus | -0.024740 | [-0.074434, 0.006282] | 1000 |
| rain_fog200 | C1:input | train | -0.039229 | [-0.168343, 0.011820] | 1000 |
| rain_fog200 | C1:input | truck | -0.030100 | [-0.056420, -0.001190] | 1000 |
| rain_fog200 | C1:input | recall | 0.013216 | [-0.000694, 0.027699] | 1000 |
| rain_fog200 | C1:input | fp_per_image | 0.010000 | [-0.100000, 0.125000] | 1000 |
| rain_fog200 | C1:input | opr | -0.047458 | [-0.067221, -0.030252] | 1000 |
| rain_fog200 | C0:input | map50 | -0.007182 | [-0.032435, 0.013445] | 1000 |
| rain_fog200 | C0:input | map50_95 | -0.009866 | [-0.030443, 0.002157] | 1000 |
| rain_fog200 | C0:input | bus | -0.024746 | [-0.077486, 0.011517] | 1000 |
| rain_fog200 | C0:input | train | -0.039855 | [-0.168342, 0.010020] | 1000 |
| rain_fog200 | C0:input | truck | -0.031592 | [-0.056485, -0.004478] | 1000 |
| rain_fog200 | C0:input | recall | 0.006608 | [-0.007504, 0.020898] | 1000 |
| rain_fog200 | C0:input | fp_per_image | 0.020000 | [-0.095000, 0.130000] | 1000 |
| rain_fog200 | C0:input | opr | -0.052542 | [-0.071810, -0.034165] | 1000 |
| rain_fog200 | C1:v4 | map50 | 0.000640 | [-0.022651, 0.014600] | 1000 |
| rain_fog200 | C1:v4 | map50_95 | -0.000640 | [-0.016414, 0.007275] | 1000 |
| rain_fog200 | C1:v4 | bus | 0.010555 | [-0.007200, 0.033844] | 1000 |
| rain_fog200 | C1:v4 | train | -0.033100 | [-0.162921, 0.032803] | 1000 |
| rain_fog200 | C1:v4 | truck | -0.017885 | [-0.036208, 0.001933] | 1000 |
| rain_fog200 | C1:v4 | recall | 0.000734 | [-0.010427, 0.010797] | 1000 |
| rain_fog200 | C1:v4 | fp_per_image | 0.035000 | [-0.050000, 0.120000] | 1000 |
| rain_fog200 | C1:v4 | opr | -0.005085 | [-0.020380, 0.009779] | 1000 |
| all406 | C1:C0 | map50 | 0.000928 | [-0.002229, 0.005456] | 1000 |
| all406 | C1:C0 | map50_95 | 0.001765 | [-0.000699, 0.003976] | 1000 |
| all406 | C1:C0 | bus | 0.002207 | [-0.014109, 0.021622] | 1000 |
| all406 | C1:C0 | train | -0.002162 | [-0.011795, 0.008738] | 1000 |
| all406 | C1:C0 | truck | 0.001681 | [-0.006300, 0.010123] | 1000 |
| all406 | C1:C0 | recall | 0.006339 | [0.002600, 0.010333] | 1000 |
| all406 | C1:C0 | fp_per_image | 0.004926 | [-0.034483, 0.041872] | 1000 |
| all406 | C1:C0 | opr | 0.004955 | [-0.001929, 0.010968] | 1000 |
| all406 | C1:input | map50 | -0.017934 | [-0.029566, -0.004328] | 1000 |
| all406 | C1:input | map50_95 | -0.010824 | [-0.017450, -0.003552] | 1000 |
| all406 | C1:input | bus | -0.021958 | [-0.055889, -0.001284] | 1000 |
| all406 | C1:input | train | -0.071320 | [-0.136580, -0.003699] | 1000 |
| all406 | C1:input | truck | -0.019552 | [-0.039561, 0.000551] | 1000 |
| all406 | C1:input | recall | 0.007084 | [-0.001135, 0.016088] | 1000 |
| all406 | C1:input | fp_per_image | 0.046798 | [-0.027155, 0.118227] | 1000 |
| all406 | C1:input | opr | -0.046581 | [-0.059884, -0.033237] | 1000 |
| all406 | C0:input | map50 | -0.018862 | [-0.031171, -0.005840] | 1000 |
| all406 | C0:input | map50_95 | -0.012589 | [-0.019196, -0.005533] | 1000 |
| all406 | C0:input | bus | -0.024164 | [-0.069298, 0.004607] | 1000 |
| all406 | C0:input | train | -0.069158 | [-0.132818, -0.003239] | 1000 |
| all406 | C0:input | truck | -0.021232 | [-0.040876, -0.002258] | 1000 |
| all406 | C0:input | recall | 0.000746 | [-0.007340, 0.009343] | 1000 |
| all406 | C0:input | fp_per_image | 0.041872 | [-0.027094, 0.110837] | 1000 |
| all406 | C0:input | opr | -0.051536 | [-0.066072, -0.037830] | 1000 |
| all406 | C1:v4 | map50 | -0.005348 | [-0.016289, 0.007992] | 1000 |
| all406 | C1:v4 | map50_95 | 0.000290 | [-0.003890, 0.007085] | 1000 |
| all406 | C1:v4 | bus | 0.010520 | [-0.001290, 0.030759] | 1000 |
| all406 | C1:v4 | train | -0.041447 | [-0.117287, 0.034873] | 1000 |
| all406 | C1:v4 | truck | -0.009402 | [-0.021291, 0.002485] | 1000 |
| all406 | C1:v4 | recall | 0.001864 | [-0.004488, 0.008588] | 1000 |
| all406 | C1:v4 | fp_per_image | 0.004926 | [-0.051724, 0.059113] | 1000 |
| all406 | C1:v4 | opr | 0.000991 | [-0.010586, 0.012733] | 1000 |
