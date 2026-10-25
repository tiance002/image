> 此文件保留阶段性研究记录；当前结论、验证集口径和公开资源范围以根目录 README 与置信度公平对照报告为准。

# 公开真实数据接入操作

更新时间：2026-10-05（Asia/Shanghai）

## 当前事实

- ACDC官方检测标注包约4MB，覆盖train/val共2006张图；匿名RGB图像包约15.6GB；
- ACDC要求注册并接受非商业许可，程序不能代替使用者同意条款；
- RTTS原官方短链当前不可达；FogGuard研究代码公布的旧Roboflow签名链接返回403；
- 新的Roboflow RTTS项目包含4322张图并声明CC BY 4.0，但下载接口需要登录/API密钥；
- 因此当前阻塞不是代码或算力，而是第三方账号与许可接受。

## 推荐下载顺序

1. 在ACDC官方页面注册并接受许可：https://acdc.vision.ee.ethz.ch/download
2. 先下载`gt_detection_trainval.zip`（约4MB）；
3. 再下载`rgb_anon_trainvaltest.zip`（约15.6GB）；
4. 不要修改压缩包文件名或内部目录；
5. 将文件保存到本机任意位置，导入程序会读取但不会删除原压缩包。

RTTS工程镜像备选：https://universe.roboflow.com/sophie-aj5fi/rtts-vf9q6-zxh7o

## 一键安全导入

ACDC标注与图像建议分别导入空目录，保留独立哈希记录：

```powershell
python scripts\import_public_dataset.py --dataset acdc --archive "<LOCAL_PATH>" --destination data\raw\ACDC_detection
python scripts\import_public_dataset.py --dataset acdc --archive "<LOCAL_PATH>" --destination data\raw\ACDC_images
```

RTTS Roboflow导出包：

```powershell
python scripts\import_public_dataset.py --dataset rtts_roboflow --archive "<LOCAL_PATH>" --destination data\raw\RTTS_roboflow_imported
```

导入器会：

- 拒绝ZIP路径穿越；
- 拒绝覆盖非空目录；
- 计算原压缩包SHA-256；
- 统计图片、XML和JSON数量；
- 写入`IMPORT_RECORD.json`保存来源、许可说明、时间和文件清单。

## 导入后的执行顺序

1. 根据`IMPORT_RECORD.json`确认实际图像根目录和标注文件；
2. COCO格式运行`evaluate_coco_detection.py --inspect-only`；
3. VOC格式运行`inspect_voc_dataset.py`；
4. 冻结类别映射和测试划分；
5. 依次评价weather input、DCP、direct、raw conservative和calibrated fusion；
6. 生成bootstrap置信区间，不根据测试结果重新调阈值。
