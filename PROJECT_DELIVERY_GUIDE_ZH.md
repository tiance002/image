> 历史离线交付说明：本公开仓库仅包含源码和汇总结果，不含本文描述的数据、权重或离线大文件包。当前研究结果与安装入口以根目录 README 为准。

# SafeRestore-RF 项目交付与运行说明

版本日期：2026-10-12  
项目名称：面向安全感知的保真型恶劣天气图像复原  
工程名称：SafeRestore-RF

## 1. 交付包包含什么

本交付包面向离线演示、继续开发和实验复现，包含：

- 完整 Python 源代码、配置、脚本、测试和 VS Code 配置；
- Gradio 单图、批量图片和离线视频演示界面；
- V1/V2/V3 训练权重及当前默认 V3 seed 42 权重；
- YOLOv8n 冻结检测器权重；
- ACDC 已解压匿名 RGB 图像和检测标注；
- 工程合成数据、全因子数据、清单和派生标注；
- ACDC 五方法评测、检测安全门控、性能基准和展示结果；
- 79 项自动化测试及项目文档。

为了避免同一数据占用两份空间，交付 ZIP **不重复包含** `data/archives/` 中的原始下载压缩包；`data/raw/` 中已解压、可直接运行的数据完整保留。Python 虚拟环境不包含在交付包中，需要在目标电脑重新建立。

## 2. 数据许可与使用边界

ACDC 数据要求用户在官方网站注册并接受许可，许可限定非商业用途。此交付包只应在已经取得许可的个人或课题组内部保存和使用：

- 不要把包含 ACDC 图像的完整 ZIP 上传到公开网盘、GitHub 或公开数据平台；
- 转交给其他人员前，应确认对方已接受 ACDC 官方许可；
- 论文或报告中使用 ACDC 时，应按官方要求引用；
- 官方说明和许可文件位于 `data/raw/ACDC_images/` 与 `data/raw/ACDC_detection/`；
- 官方入口：<https://acdc.vision.ee.ethz.ch/>。

本项目当前 ACDC 结果来自冻结验证集探索，只能支持“检测安全门控基本消除复原造成的检测损失”，不能宣称显著提升自动驾驶安全或检测性能。

## 3. 当前数据规模

不计重复的原始下载 ZIP，`data/` 约 15.65 GB、8322 个文件。ACDC RGB 图像按官方条件和划分保存：

| 条件 | train | val | test | train_ref | val_ref | test_ref |
|---|---:|---:|---:|---:|---:|---:|
| fog | 400 | 100 | 500 | 400 | 100 | 500 |
| rain | 400 | 100 | 500 | 400 | 100 | 500 |
| night | 400 | 106 | 500 | 400 | 106 | 500 |
| snow | 400 | 100 | 500 | 400 | 100 | 500 |

当前研究主线只使用 fog 和 rain。night、snow 和对应清晰参考图保留在数据包中，供后续扩展，不进入当前主结论。

主要目录：

```text
data/
├── raw/ACDC_images/               # ACDC 已解压 RGB 图像
├── raw/ACDC_detection/            # ACDC COCO 检测标注与许可说明
├── derived/ACDC_detection/        # 确定性生成的 fog/rain 子集标注
├── factorial/                     # 3天气×3强度全因子工程数据
├── manifests/                     # 训练/验证/测试清单
└── processed/                     # 批量复原和传统基线结果
```

## 4. 推荐运行环境

已验证环境：

- Windows 10/11；
- Python 3.12.7；
- NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）；
- PyTorch 2.13.0+cu126，CUDA 12.6；
- Gradio 6.29.1；
- Ultralytics 8.4.170；
- OpenCV 5.0.0；
- NumPy 2.5.2。

推荐 Python 3.10–3.12。没有 NVIDIA GPU 时可以尝试 CPU，但速度会明显下降，视频演示可能无法达到可接受帧率。

## 5. 解压后在 VS Code 打开

1. 解压交付 ZIP，保持目录层级不变；
2. 打开 VS Code；
3. 选择“文件 → 打开文件夹”；
4. 选择解压后的 `数字图像处理` 文件夹；
5. 选择“终端 → 新建终端”。

不要只打开 `app.py`，必须打开整个项目文件夹，否则相对路径、权重和数据目录可能无法识别。

## 6. 在新电脑建立环境

在项目终端执行：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,metrics,detection,demo]"
```

如果需要 CUDA，请先按照 PyTorch 官方页面选择与目标电脑显卡驱动匹配的安装命令，再执行最后一行项目安装命令。

当前电脑也可以继续使用已配置解释器：

```text
<LOCAL_PATH>
```

该绝对路径只适用于当前电脑，不要写入另一台电脑的配置。

## 7. 最快启动方式

激活环境后，在项目根目录执行：

```powershell
python app.py
```

终端出现以下内容后，在浏览器打开对应地址：

```text
Running on local URL: http://127.0.0.1:7860
```

端口占用时会自动选择 7861、7862 等后续端口，以终端实际输出为准。终端必须保持运行；停止服务请按 `Ctrl+C`。

也可以在 VS Code 中按 `F5`，选择 `SafeRestore: 启动上传演示`。

## 8. 网页功能操作

### 8.1 单图实验

1. 上传 JPG/PNG，或使用默认雾天样例；
2. 点击“开始复原与安全检测”；
3. 查看安全融合、道路目标检测、直接复原、DCP、不确定性和候选改动热力图；
4. 查看复原/DCP/检测门控/总链路耗时；
5. 下载单图完整实验包 ZIP。

单图 ZIP 包含输入、直接复原、候选/最终输出、风险热力图、改动热力图、检测结果、`metrics.json` 和带 SHA-256 的 `manifest.json`。

### 8.2 批量图片实验

1. 一次上传 1–12 张道路图片；
2. 点击“开始批量复原与安全检测”；
3. 查看逐图雨/雾概率、强度、安全分数、采用/回退、目标数和耗时；
4. 下载批量总 ZIP 和汇总 CSV。

批量总 ZIP 的 `items/` 中包含每张图片的完整单图 ZIP。命令行自检：

```powershell
python app.py --check-batch
```

### 8.3 无摄像头视频仿真

1. 上传 MP4，或使用预载 ACDC 雾天序列；
2. 首次建议使用 30 帧、10 FPS、640 最大边、0.18 时序强度；
3. 保持“每帧评估”和 FP32 作为安全基线；
4. 点击“开始视频仿真”；
5. 查看结果视频、安全时间轴、P50/P95 延迟、丢帧仿真和三级选择统计；
6. 下载视频结果总 ZIP、JSON、CSV 和时间轴。

视频自检：

```powershell
python app.py --check-video
```

## 9. 命令行自检与测试

模型和单图链路：

```powershell
python app.py --check
```

运行全部测试：

```powershell
python -m pytest -q
python -m ruff check app.py src scripts tests
python -m compileall -q app.py src scripts tests
```

交付时基线为 79 项测试全部通过。

## 10. ACDC 真实数据评测

检查数据就绪状态：

```powershell
python scripts\prepare_acdc_benchmark.py
```

运行冻结五方法评测：

```powershell
python scripts\run_acdc_benchmark.py
```

重新评价安全门控阈值：

```powershell
python scripts\evaluate_perception_guard.py --resamples 100
```

当前冻结验证集结果：输入 mAP50 为 0.2042，未门控复原候选为 0.1235，门控后为 0.2052。门控相对输入的置信区间跨越 0，因此不能写成“显著提升”。

## 11. 训练与复现实验

随机张量冒烟训练：

```powershell
python -m saferestore.train --config configs\baseline.yaml --smoke-test
```

V3 全因子 seed 42：

```powershell
python -m saferestore.train --config configs\v3_factorial_seed42.yaml --train-paired
```

多随机种子和强度鲁棒性：

```powershell
python scripts\evaluate_multiseed.py
python scripts\evaluate_severity_robustness.py
```

默认网页权重为：

```text
runs/v3_factorial_seed42/best.pt
```

检测器权重为项目根目录下的 `yolov8n.pt`。

## 12. 输出目录

```text
runs/
├── image_demo/                    # 单图结果和完整实验包
├── batch_demo/                    # 批量汇总 CSV 和总 ZIP
├── video_demo/                    # 视频、时间轴和逐帧记录
├── acdc_benchmark/                # ACDC 五方法及安全门控评测
├── performance/                   # 视频模式和精度性能基准
├── demo_report/                   # 展示报告
├── detection_report/              # 检测保真报告
└── v3_factorial_seed42/           # 当前默认模型与校准参数
```

## 13. 指标解释

- PSNR/SSIM：需要清晰真值，只能在配对数据上解释；
- 不确定性：模型对复原可靠性的相对估计，不是事故概率；
- 改动热力图：候选相对输入改动的位置，不是无参考画质评分；
- 检测安全分数：冻结检测器证据保持程度，不代表真实自动驾驶系统安全等级；
- 回退输入：候选没有达到冻结阈值时保留原始画面；
- 实时丢帧：由离线逐帧实测耗时进行的单工作线程调度仿真，不是已接入真实摄像头。

## 14. 当前已知限制

- 还没有接入真实摄像头、RTSP 或车辆 CAN 总线；
- 当前复原模型主要由工程合成数据训练；
- ACDC 当前只形成验证集探索，不是独立测试集结论；
- FP16 在当前 RTX 4060 上仅约 1.010 倍加速，因此默认仍为 FP32；
- 视频时序稳定是工程光流模块，不是端到端视频复原网络；
- YOLOv8n 只用于冻结感知代理，不能替代自动驾驶量产感知栈。

## 15. 常见问题

### 端口被占用

项目会自动选择下一个端口。不要固定只访问 7860，应查看终端显示的 URL。

### 页面打不开

确认运行 `app.py` 的终端没有关闭，并确认浏览器访问的是 `127.0.0.1` 而不是其他设备 IP。

### CUDA 内存不足

关闭其他 GPU 程序，视频最大边先选 480 或 640，帧数先选 10–30。

### 新电脑找不到模型

确认解压后仍存在 `yolov8n.pt`、`runs/v3_factorial_seed42/best.pt` 和 `fusion_calibration.json`，且启动终端位于项目根目录。

### 数据路径显示旧电脑绝对路径

部分 `IMPORT_RECORD.json` 和历史报告保存的是导入时审计路径，仅用于追溯；运行脚本使用项目内相对目录，不需要手工修改历史记录。

## 16. 完整包校验

交付目录中会同时生成 `.sha256.txt`。复制或解压前可在 PowerShell 执行：

```powershell
Get-FileHash -Algorithm SHA256 ".\SafeRestore-RF_full_project_with_datasets_2026-10-12.zip"
```

将输出哈希与 `.sha256.txt` 中记录的值比较。两者一致说明大文件传输过程中没有损坏。

## 17. 重新打包

项目提供可重复执行的脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_release_package.ps1
```

脚本保留已解压数据，排除重复的 `data/archives/`、Python 缓存和测试缓存，并在项目同级的 `release_packages` 目录生成 ZIP、SHA-256 和元数据 JSON。英文包名用于兼容 Windows PowerShell 5.1。
