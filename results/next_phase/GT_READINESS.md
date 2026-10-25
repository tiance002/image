# 真实检测监督资源与实现核查

状态：本地真实 ACDC train 数据可用，P1 训练不因缺少 GT 阻塞。下列反传只属于实现冒烟，不属于 C0/C1 训练结果。

## 数据范围

- 原始标注：`data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json`。
- 标注 SHA256：`33be9416b6c4c66d55fe61ee0d3dd1e2ecf06fc82db38480aa377e5341c24487`。
- 图像根：`data/raw/ACDC_images/rgb_anon`；1600 个图像路径全部在本地存在，fog/rain/night/snow 各400张。
- 共10135条框标注。保留七类非crowd框9607条：person1657、bicycle344、car6450、motorcycle239、bus168、train212、truck537。
- 528条进入训练忽略区域：304条crowd，加224条非crowd rider。rider 不等价于 person，不强行迁移类别。
- ACDC类别24/26/27/28/31/32/33映射为COCO零基类别0/2/7/5/6/3/1。
- `rain/instancesonly_rain_train_gt_detection.json` 实际是1600张全天气combined，与顶层train文件字节相同，不得将其再次拼接到分天气清单。
- 数据集代码拒绝 `/val/`、`/test/`、`/train_ref/`、路径越界及重复文件。真实框来自同一恶劣天气帧，未迁移参考帧标注。

## 划分局限

官方train1600与val406的图像文件名及basename无交集，但按weather/sequence键有15组重合：

`fog/GOPR0476`, `fog/GP010476`, `fog/GP020475`, `night/GOPR0351`, `rain/GOPR0400`, `rain/GOPR0402`, `rain/GP010400`, `rain/GP020402`, `rain/GP030400`, `snow/GOPR0122`, `snow/GOPR0604`, `snow/GOPR0606`, `snow/GOPR0607`, `snow/GP010607`, `snow/GP030176`。

这不是validation图像参与反传，但不能称为场景独立或独立测试。本轮沿用官方划分并披露限制，不临时按validation得分删选train场景。历史v4的配对训练也来自ACDC train相关场景。

## 原生损失复用边界

使用本机Ultralytics8.4.104 `v8DetectionLoss`，保留 `TaskAlignedAssigner`、`BboxLoss` 和DFL原实现，原生gain为box7.5、cls0.5、dfl1.5。保持80维COCO头及所有权重不变，仅在criterion的class_weights中屏蔽未标注的73类。

冻结独立加载的YOLO实例并保持eval/BatchNorm固定，学生前向保留梯度。crowd/rider的训练忽略策略为：anchor中心在忽略框内，或当前预测框与忽略框有任何正面积相交时，不计算该anchor的负分类监督；已有原生匹配正监督保留。该策略保守地减少负样本，不是COCO crowd评估算法，不改写原生任务匹配、box回归或DFL。

图像预处理使用RGB、完整视野640方形letterbox；GT按实际整数缩放尺寸同步缩放/补边，默认不翻转。可选水平翻转同步变换框并有专项测试。

## 实际验证

- `tests/test_task_detection.py` 8项通过：RGB/框letterbox与翻转、crowd交集与跨batch隔离、忽略区域正监督保留、类别映射及重复数据拒绝、非train拒绝、缺权重禁止下载、真实本地YOLO反传及状态不变。
- 测试直接检查未标注73类的raw-logit梯度为0，七类梯度存在；检测器所有参数无grad，完整state_dict（含BN统计）不变。
- 对真实train bus帧 `fog/train/GP020478/GP020478_frame_000142_rgb_anon.png` 的CPU640单次反传：box=0.895131、cls=0.710538、DFL=0.992324、总损失2.597994；输入梯度L2=7.655294且全部有限，检测器参数梯度数=0。见 `gt_loss_smoke.json`。
- 全程本地数据/本地权重，没有安装或下载。
