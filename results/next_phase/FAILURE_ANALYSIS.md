# bus/train 逐例失败分析

本报告只读取已保存预测；不重新推理、不使用验证 GT 训练。“命中/FP”来自官方 COCOeval 在 IoU=0.5 的逐图一对一匹配，crowd 匹配被忽略。

## 证据口径

- 运行时以独立 conf=0.25 输出、maxDets=300 判定检出；AP/PR 以 conf=0.001、maxDets=100 统计。
- GT 原始输入漏检与复原新增损失分开；分类允许多标签，分类计数不能相加作为唯一目标数。
- 置信度下降须看到同类 IoU≥0.5 的低分框；空匹配本身不能证明置信度下降。
- 类别错误与定位类别是有阈值证据的候选解释，不证明预测框与该物体具有唯一身份；重复框或邻近物体仍可能造成歧义。
- NMS 证据须存在同类 pre-NMS 高分正确候选及更高分 post-NMS 抑制框，二者 IoU>配置阈值；排除近似同框 survivor。保存的是 decoded 坐标，缺少原始 NMS 索引；边界坐标舍入/letterbox 逆变换可有小偏差，因此称“可复核 NMS 证据”，不声称已追踪算子内部抑制索引。
- pre-NMS 只保存最大类别候选，未保留每个 anchor 所有非最大类别 logits；不能由它排除全部类别竞争。
- 新增高排序 FP 必须被官方匹配判为非忽略 FP、在 input 中无同类 IoU≥0.5 的对应 FP，并在该天气/类别排序中排在至少一个 TP 前面；并不证明删掉单个框后的 AP 反事实增量。

## 全部非 crowd bus/train GT 的运行时检出

| 范围 | 类别 | GT | input | v4 | guard | C0 | C1 |
|---|---|---:|---:|---:|---:|---:|---:|
| all406 | bus | 49 | 9 | 6 | 5 | 6 | 6 |
| all406 | train | 46 | 18 | 17 | 17 | 16 | 16 |
| rain_fog200 | bus | 37 | 8 | 6 | 5 | 6 | 6 |
| rain_fog200 | train | 12 | 4 | 4 | 4 | 3 | 3 |
| rain | bus | 32 | 5 | 3 | 3 | 3 | 3 |
| rain | train | 9 | 2 | 2 | 2 | 2 | 2 |

## 解释标签数量（多标签，可重叠）

| 方法 | 范围 | 类别 | 标签 | 数量 |
|---|---|---|---|---:|
| C0 | all406 | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C0 | all406 | bus | 正确类别高分框定位不足 | 4 |
| C0 | all406 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| C0 | all406 | bus | 原始输入已漏检 | 40 |
| C0 | all406 | bus | 现有证据无法区分 | 32 |
| C0 | all406 | bus | 与 GT 重叠的其他类别候选 | 5 |
| C0 | all406 | train | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C0 | all406 | train | 正确类别高分框定位不足 | 1 |
| C0 | all406 | train | 满足候选和抑制框条件的 NMS 证据 | 1 |
| C0 | all406 | train | 原始输入已漏检 | 28 |
| C0 | all406 | train | 现有证据无法区分 | 24 |
| C0 | all406 | train | 与 GT 重叠的其他类别候选 | 4 |
| C0 | rain | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C0 | rain | bus | 正确类别高分框定位不足 | 3 |
| C0 | rain | bus | 新增且排在真实 TP 前面的 FP | 1 |
| C0 | rain | bus | 原始输入已漏检 | 27 |
| C0 | rain | bus | 现有证据无法区分 | 20 |
| C0 | rain | bus | 与 GT 重叠的其他类别候选 | 4 |
| C0 | rain | train | 原始输入已漏检 | 7 |
| C0 | rain | train | 现有证据无法区分 | 6 |
| C0 | rain | train | 与 GT 重叠的其他类别候选 | 1 |
| C0 | rain_fog200 | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C0 | rain_fog200 | bus | 正确类别高分框定位不足 | 3 |
| C0 | rain_fog200 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| C0 | rain_fog200 | bus | 原始输入已漏检 | 29 |
| C0 | rain_fog200 | bus | 现有证据无法区分 | 21 |
| C0 | rain_fog200 | bus | 与 GT 重叠的其他类别候选 | 5 |
| C0 | rain_fog200 | train | 正确类别但置信度降低并跨过运行时阈值 | 1 |
| C0 | rain_fog200 | train | 原始输入已漏检 | 8 |
| C0 | rain_fog200 | train | 现有证据无法区分 | 7 |
| C0 | rain_fog200 | train | 与 GT 重叠的其他类别候选 | 1 |
| C1 | all406 | bus | 正确类别但置信度降低并跨过运行时阈值 | 3 |
| C1 | all406 | bus | 正确类别高分框定位不足 | 4 |
| C1 | all406 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| C1 | all406 | bus | 原始输入已漏检 | 40 |
| C1 | all406 | bus | 现有证据无法区分 | 31 |
| C1 | all406 | bus | 与 GT 重叠的其他类别候选 | 5 |
| C1 | all406 | train | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C1 | all406 | train | 正确类别高分框定位不足 | 1 |
| C1 | all406 | train | 满足候选和抑制框条件的 NMS 证据 | 1 |
| C1 | all406 | train | 原始输入已漏检 | 28 |
| C1 | all406 | train | 现有证据无法区分 | 25 |
| C1 | all406 | train | 与 GT 重叠的其他类别候选 | 3 |
| C1 | rain | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C1 | rain | bus | 正确类别高分框定位不足 | 3 |
| C1 | rain | bus | 新增且排在真实 TP 前面的 FP | 1 |
| C1 | rain | bus | 原始输入已漏检 | 27 |
| C1 | rain | bus | 现有证据无法区分 | 20 |
| C1 | rain | bus | 与 GT 重叠的其他类别候选 | 4 |
| C1 | rain | train | 原始输入已漏检 | 7 |
| C1 | rain | train | 现有证据无法区分 | 6 |
| C1 | rain | train | 与 GT 重叠的其他类别候选 | 1 |
| C1 | rain_fog200 | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| C1 | rain_fog200 | bus | 正确类别高分框定位不足 | 3 |
| C1 | rain_fog200 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| C1 | rain_fog200 | bus | 原始输入已漏检 | 29 |
| C1 | rain_fog200 | bus | 现有证据无法区分 | 21 |
| C1 | rain_fog200 | bus | 与 GT 重叠的其他类别候选 | 5 |
| C1 | rain_fog200 | train | 正确类别但置信度降低并跨过运行时阈值 | 1 |
| C1 | rain_fog200 | train | 原始输入已漏检 | 8 |
| C1 | rain_fog200 | train | 现有证据无法区分 | 7 |
| C1 | rain_fog200 | train | 与 GT 重叠的其他类别候选 | 1 |
| guard | all406 | bus | 正确类别但置信度降低并跨过运行时阈值 | 4 |
| guard | all406 | bus | 正确类别高分框定位不足 | 5 |
| guard | all406 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| guard | all406 | bus | 原始输入已漏检 | 40 |
| guard | all406 | bus | 现有证据无法区分 | 29 |
| guard | all406 | bus | 与 GT 重叠的其他类别候选 | 7 |
| guard | all406 | train | 原始输入已漏检 | 28 |
| guard | all406 | train | 现有证据无法区分 | 26 |
| guard | all406 | train | 与 GT 重叠的其他类别候选 | 3 |
| guard | rain | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| guard | rain | bus | 正确类别高分框定位不足 | 4 |
| guard | rain | bus | 新增且排在真实 TP 前面的 FP | 1 |
| guard | rain | bus | 原始输入已漏检 | 27 |
| guard | rain | bus | 现有证据无法区分 | 18 |
| guard | rain | bus | 与 GT 重叠的其他类别候选 | 5 |
| guard | rain | train | 原始输入已漏检 | 7 |
| guard | rain | train | 现有证据无法区分 | 7 |
| guard | rain_fog200 | bus | 正确类别但置信度降低并跨过运行时阈值 | 3 |
| guard | rain_fog200 | bus | 正确类别高分框定位不足 | 4 |
| guard | rain_fog200 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| guard | rain_fog200 | bus | 原始输入已漏检 | 29 |
| guard | rain_fog200 | bus | 现有证据无法区分 | 19 |
| guard | rain_fog200 | bus | 与 GT 重叠的其他类别候选 | 7 |
| guard | rain_fog200 | train | 原始输入已漏检 | 8 |
| guard | rain_fog200 | train | 现有证据无法区分 | 8 |
| v4 | all406 | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| v4 | all406 | bus | 正确类别高分框定位不足 | 3 |
| v4 | all406 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| v4 | all406 | bus | 原始输入已漏检 | 40 |
| v4 | all406 | bus | 现有证据无法区分 | 34 |
| v4 | all406 | bus | 与 GT 重叠的其他类别候选 | 4 |
| v4 | all406 | train | 原始输入已漏检 | 28 |
| v4 | all406 | train | 现有证据无法区分 | 25 |
| v4 | all406 | train | 与 GT 重叠的其他类别候选 | 4 |
| v4 | rain | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| v4 | rain | bus | 正确类别高分框定位不足 | 2 |
| v4 | rain | bus | 新增且排在真实 TP 前面的 FP | 1 |
| v4 | rain | bus | 原始输入已漏检 | 27 |
| v4 | rain | bus | 现有证据无法区分 | 22 |
| v4 | rain | bus | 与 GT 重叠的其他类别候选 | 3 |
| v4 | rain | train | 原始输入已漏检 | 7 |
| v4 | rain | train | 现有证据无法区分 | 6 |
| v4 | rain | train | 与 GT 重叠的其他类别候选 | 1 |
| v4 | rain_fog200 | bus | 正确类别但置信度降低并跨过运行时阈值 | 2 |
| v4 | rain_fog200 | bus | 正确类别高分框定位不足 | 2 |
| v4 | rain_fog200 | bus | 新增且排在真实 TP 前面的 FP | 1 |
| v4 | rain_fog200 | bus | 原始输入已漏检 | 29 |
| v4 | rain_fog200 | bus | 现有证据无法区分 | 23 |
| v4 | rain_fog200 | bus | 与 GT 重叠的其他类别候选 | 4 |
| v4 | rain_fog200 | train | 原始输入已漏检 | 8 |
| v4 | rain_fog200 | train | 现有证据无法区分 | 7 |
| v4 | rain_fog200 | train | 与 GT 重叠的其他类别候选 | 1 |

## AP50 新协议实测

同一新协议下比较，不能直接与历史 0.25 截断 AP 的差值混用。

| 方法 | 范围 | bus AP50 | train AP50 | truck AP50 |
|---|---|---:|---:|---:|
| input | all406 | 0.121324 | 0.382810 | 0.159388 |
| input | rain_fog200 | 0.173570 | 0.323432 | 0.149280 |
| input | fog | 0.465347 | 0.663366 | 0.126395 |
| input | rain | 0.139467 | 0.242574 | 0.250716 |
| input | night | 未提供 | 0.298644 | 0.141715 |
| input | snow | 0.037286 | 0.681931 | 0.243566 |
| v4 | all406 | 0.088846 | 0.352937 | 0.149238 |
| v4 | rain_fog200 | 0.138274 | 0.317303 | 0.137066 |
| v4 | fog | 0.490806 | 0.663366 | 0.117161 |
| v4 | rain | 0.092430 | 0.235502 | 0.221916 |
| v4 | night | 未提供 | 0.244043 | 0.150200 |
| v4 | snow | 0.005179 | 0.681931 | 0.232053 |
| guard | all406 | 0.094568 | 0.346211 | 0.150943 |
| guard | rain_fog200 | 0.137173 | 0.319891 | 0.136394 |
| guard | fog | 0.465347 | 0.663366 | 0.121286 |
| guard | rain | 0.092283 | 0.237933 | 0.211695 |
| guard | night | 未提供 | 0.229906 | 0.151101 |
| guard | snow | 0.015713 | 0.681931 | 0.231796 |
| C0 | all406 | 0.097160 | 0.313652 | 0.138155 |
| C0 | rain_fog200 | 0.148824 | 0.283577 | 0.117688 |
| C0 | fog | 0.490806 | 0.429986 | 0.111094 |
| C0 | rain | 0.119902 | 0.242574 | 0.213325 |
| C0 | night | 未提供 | 0.199749 | 0.141687 |
| C0 | snow | 0.002122 | 0.681931 | 0.237989 |
| C1 | all406 | 0.099366 | 0.311490 | 0.139836 |
| C1 | rain_fog200 | 0.148829 | 0.284203 | 0.119180 |
| C1 | fog | 0.480198 | 0.467327 | 0.109231 |
| C1 | rain | 0.114109 | 0.243281 | 0.219753 |
| C1 | night | 未提供 | 0.198664 | 0.142082 |
| C1 | snow | 0.015248 | 0.669554 | 0.236399 |

## 正确类别但置信度降低并跨过运行时阈值

最多展示 3 个真实样例，优先雨天 bus。可用样例 3 个。

- `rain/val/GOPR0402/GOPR0402_frame_000408_rgb_anon.png`；image_id=214，GT/候选类别=bus，分析方法=C0，GT id=1269。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | bus / 0.3678 / IoU 0.5449 | bus / 0.3678 / IoU 0.5449 | 无匹配 |
| v4 | 无匹配 | bus / 0.1587 / IoU 0.5446 | 无匹配 |
| guard | 无匹配 | bus / 0.1660 / IoU 0.5424 | 无匹配 |
| C0 | 无匹配 | bus / 0.2139 / IoU 0.5426 | 无匹配 |
| C1 | 无匹配 | bus / 0.1994 / IoU 0.5414 | 无匹配 |

证据：`{"input": {"category_id": 5, "category": "bus", "confidence": 0.36784136295318604, "box": "coordinates omitted from source snapshot", "iou": 0.544938001026584}, "candidate": {"category_id": 5, "category": "bus", "confidence": 0.21386948227882385, "box": "coordinates omitted from source snapshot", "iou": 0.5426234081120997}}`

- `rain/val/GOPR0402/GOPR0402_frame_000408_rgb_anon.png`；image_id=214，GT/候选类别=bus，分析方法=C1，GT id=1269。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | bus / 0.3678 / IoU 0.5449 | bus / 0.3678 / IoU 0.5449 | 无匹配 |
| v4 | 无匹配 | bus / 0.1587 / IoU 0.5446 | 无匹配 |
| guard | 无匹配 | bus / 0.1660 / IoU 0.5424 | 无匹配 |
| C0 | 无匹配 | bus / 0.2139 / IoU 0.5426 | 无匹配 |
| C1 | 无匹配 | bus / 0.1994 / IoU 0.5414 | 无匹配 |

证据：`{"input": {"category_id": 5, "category": "bus", "confidence": 0.36784136295318604, "box": "coordinates omitted from source snapshot", "iou": 0.544938001026584}, "candidate": {"category_id": 5, "category": "bus", "confidence": 0.19944408535957336, "box": "coordinates omitted from source snapshot", "iou": 0.5413830026932391}}`

- `rain/val/GOPR0402/GOPR0402_frame_000408_rgb_anon.png`；image_id=214，GT/候选类别=bus，分析方法=guard，GT id=1269。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | bus / 0.3678 / IoU 0.5449 | bus / 0.3678 / IoU 0.5449 | 无匹配 |
| v4 | 无匹配 | bus / 0.1587 / IoU 0.5446 | 无匹配 |
| guard | 无匹配 | bus / 0.1660 / IoU 0.5424 | 无匹配 |
| C0 | 无匹配 | bus / 0.2139 / IoU 0.5426 | 无匹配 |
| C1 | 无匹配 | bus / 0.1994 / IoU 0.5414 | 无匹配 |

证据：`{"input": {"category_id": 5, "category": "bus", "confidence": 0.36784136295318604, "box": "coordinates omitted from source snapshot", "iou": 0.544938001026584}, "candidate": {"category_id": 5, "category": "bus", "confidence": 0.16595937311649323, "box": "coordinates omitted from source snapshot", "iou": 0.5424433108820012}}`

## 与 GT 重叠的其他类别候选

最多展示 3 个真实样例，优先雨天 bus。可用样例 3 个。

- `rain/val/GOPR0402/GOPR0402_frame_000120_rgb_anon.png`；image_id=230，GT/候选类别=bus，分析方法=C0，GT id=1389。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | 无匹配 | truck / 0.6221 / IoU 0.9698 |
| v4 | 无匹配 | 无匹配 | truck / 0.5389 / IoU 0.9584 |
| guard | 无匹配 | 无匹配 | truck / 0.5931 / IoU 0.9694 |
| C0 | 无匹配 | 无匹配 | truck / 0.6433 / IoU 0.9639 |
| C1 | 无匹配 | 无匹配 | truck / 0.6454 / IoU 0.9667 |

证据：`{"category_id": 7, "category": "truck", "confidence": 0.643318772315979, "box": "coordinates omitted from source snapshot", "iou": 0.9638650030303152}`

- `rain/val/GOPR0402/GOPR0402_frame_000120_rgb_anon.png`；image_id=230，GT/候选类别=bus，分析方法=C1，GT id=1389。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | 无匹配 | truck / 0.6221 / IoU 0.9698 |
| v4 | 无匹配 | 无匹配 | truck / 0.5389 / IoU 0.9584 |
| guard | 无匹配 | 无匹配 | truck / 0.5931 / IoU 0.9694 |
| C0 | 无匹配 | 无匹配 | truck / 0.6433 / IoU 0.9639 |
| C1 | 无匹配 | 无匹配 | truck / 0.6454 / IoU 0.9667 |

证据：`{"category_id": 7, "category": "truck", "confidence": 0.6453580856323242, "box": "coordinates omitted from source snapshot", "iou": 0.9667012062598975}`

- `rain/val/GOPR0402/GOPR0402_frame_000120_rgb_anon.png`；image_id=230，GT/候选类别=bus，分析方法=guard，GT id=1389。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | 无匹配 | truck / 0.6221 / IoU 0.9698 |
| v4 | 无匹配 | 无匹配 | truck / 0.5389 / IoU 0.9584 |
| guard | 无匹配 | 无匹配 | truck / 0.5931 / IoU 0.9694 |
| C0 | 无匹配 | 无匹配 | truck / 0.6433 / IoU 0.9639 |
| C1 | 无匹配 | 无匹配 | truck / 0.6454 / IoU 0.9667 |

证据：`{"category_id": 7, "category": "truck", "confidence": 0.5931354761123657, "box": "coordinates omitted from source snapshot", "iou": 0.9693959145301186}`

## 正确类别高分框定位不足

最多展示 3 个真实样例，优先雨天 bus。可用样例 3 个。

- `rain/val/GP020402/GP020402_frame_000802_rgb_anon.png`；image_id=264，GT/候选类别=bus，分析方法=C0，GT id=1626。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.3446 / IoU 0.4534 | 无匹配 |
| v4 | 无匹配 | bus / 0.0120 / IoU 0.4489 | 无匹配 |
| guard | 无匹配 | bus / 0.3243 / IoU 0.4694 | 无匹配 |
| C0 | 无匹配 | bus / 0.3262 / IoU 0.4654 | 无匹配 |
| C1 | 无匹配 | bus / 0.3242 / IoU 0.4663 | 无匹配 |

证据：`{"category_id": 5, "category": "bus", "confidence": 0.326174795627594, "box": "coordinates omitted from source snapshot", "iou": 0.4654403012472189}`

- `rain/val/GP020402/GP020402_frame_000802_rgb_anon.png`；image_id=264，GT/候选类别=bus，分析方法=C1，GT id=1626。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.3446 / IoU 0.4534 | 无匹配 |
| v4 | 无匹配 | bus / 0.0120 / IoU 0.4489 | 无匹配 |
| guard | 无匹配 | bus / 0.3243 / IoU 0.4694 | 无匹配 |
| C0 | 无匹配 | bus / 0.3262 / IoU 0.4654 | 无匹配 |
| C1 | 无匹配 | bus / 0.3242 / IoU 0.4663 | 无匹配 |

证据：`{"category_id": 5, "category": "bus", "confidence": 0.3241844177246094, "box": "coordinates omitted from source snapshot", "iou": 0.466251837005085}`

- `rain/val/GP020402/GP020402_frame_000802_rgb_anon.png`；image_id=264，GT/候选类别=bus，分析方法=guard，GT id=1626。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.3446 / IoU 0.4534 | 无匹配 |
| v4 | 无匹配 | bus / 0.0120 / IoU 0.4489 | 无匹配 |
| guard | 无匹配 | bus / 0.3243 / IoU 0.4694 | 无匹配 |
| C0 | 无匹配 | bus / 0.3262 / IoU 0.4654 | 无匹配 |
| C1 | 无匹配 | bus / 0.3242 / IoU 0.4663 | 无匹配 |

证据：`{"category_id": 5, "category": "bus", "confidence": 0.3243233859539032, "box": "coordinates omitted from source snapshot", "iou": 0.4694194205744142}`

## 满足候选和抑制框条件的 NMS 证据

最多展示 3 个真实样例，优先雨天 bus。可用样例 2 个。

- `night/val/GOPR0351/GOPR0351_frame_000081_rgb_anon.png`；image_id=153，GT/候选类别=train，分析方法=C0，GT id=931。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | train / 0.6208 / IoU 0.6276 | train / 0.6208 / IoU 0.6276 | 无匹配 |
| v4 | train / 0.8257 / IoU 0.5217 | train / 0.0011 / IoU 0.5771 | 无匹配 |
| guard | train / 0.8208 / IoU 0.5162 | train / 0.0021 / IoU 0.5776 | 无匹配 |
| C0 | 无匹配 | train / 0.0044 / IoU 0.5794 | 无匹配 |
| C1 | 无匹配 | train / 0.0045 / IoU 0.5792 | 无匹配 |

证据：`{"candidate": {"category_id": 6, "category": "train", "confidence": 0.7276391386985779, "box": "coordinates omitted from source snapshot", "iou": 0.5001924082365448}, "suppressor": {"category_id": 6, "category": "train", "confidence": 0.8027181029319763, "box": "coordinates omitted from source snapshot", "iou": 0.490166099403743}, "candidate_suppressor_iou": 0.9500896965135661, "qualification": "Decoded-coordinate evidence under configured class-aware NMS; original indices are not saved."}`

- `night/val/GOPR0351/GOPR0351_frame_000081_rgb_anon.png`；image_id=153，GT/候选类别=train，分析方法=C1，GT id=931。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | train / 0.6208 / IoU 0.6276 | train / 0.6208 / IoU 0.6276 | 无匹配 |
| v4 | train / 0.8257 / IoU 0.5217 | train / 0.0011 / IoU 0.5771 | 无匹配 |
| guard | train / 0.8208 / IoU 0.5162 | train / 0.0021 / IoU 0.5776 | 无匹配 |
| C0 | 无匹配 | train / 0.0044 / IoU 0.5794 | 无匹配 |
| C1 | 无匹配 | train / 0.0045 / IoU 0.5792 | 无匹配 |

证据：`{"candidate": {"category_id": 6, "category": "train", "confidence": 0.7233810424804688, "box": "coordinates omitted from source snapshot", "iou": 0.5016007698507293}, "suppressor": {"category_id": 6, "category": "train", "confidence": 0.8024349212646484, "box": "coordinates omitted from source snapshot", "iou": 0.4970591774823739}, "candidate_suppressor_iou": 0.9897392569542026, "qualification": "Decoded-coordinate evidence under configured class-aware NMS; original indices are not saved."}`

## 原始输入已漏检

最多展示 3 个真实样例，优先雨天 bus。可用样例 3 个。

- `rain/val/GP030400/GP030400_frame_000129_rgb_anon.png`；image_id=206，GT/候选类别=bus，分析方法=C0，GT id=1202。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.0014 / IoU 0.4910 | 无匹配 |
| v4 | 无匹配 | 无匹配 | 无匹配 |
| guard | 无匹配 | bus / 0.0012 / IoU 0.5232 | 无匹配 |
| C0 | 无匹配 | 无匹配 | 无匹配 |
| C1 | 无匹配 | 无匹配 | 无匹配 |

证据：`{"input_hit": false, "candidate_hit": false}`

- `rain/val/GP030400/GP030400_frame_000129_rgb_anon.png`；image_id=206，GT/候选类别=bus，分析方法=C1，GT id=1202。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.0014 / IoU 0.4910 | 无匹配 |
| v4 | 无匹配 | 无匹配 | 无匹配 |
| guard | 无匹配 | bus / 0.0012 / IoU 0.5232 | 无匹配 |
| C0 | 无匹配 | 无匹配 | 无匹配 |
| C1 | 无匹配 | 无匹配 | 无匹配 |

证据：`{"input_hit": false, "candidate_hit": false}`

- `rain/val/GP030400/GP030400_frame_000129_rgb_anon.png`；image_id=206，GT/候选类别=bus，分析方法=guard，GT id=1202。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.0014 / IoU 0.4910 | 无匹配 |
| v4 | 无匹配 | 无匹配 | 无匹配 |
| guard | 无匹配 | bus / 0.0012 / IoU 0.5232 | 无匹配 |
| C0 | 无匹配 | 无匹配 | 无匹配 |
| C1 | 无匹配 | 无匹配 | 无匹配 |

证据：`{"input_hit": false, "candidate_hit": false}`

## 新增且排在真实 TP 前面的 FP

最多展示 3 个真实样例，优先雨天 bus。可用样例 3 个。

- `rain/val/GOPR0572/GOPR0572_frame_000825_rgb_anon.png`；image_id=242，GT/候选类别=bus，分析方法=C0，GT id=None。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|

证据：`{"image_id": 242, "category_id": 5, "confidence": 0.3712635040283203, "box": "coordinates omitted from source snapshot", "matched_gt_id": null, "file_name": "rain/val/GOPR0572/GOPR0572_frame_000825_rgb_anon.png", "weather": "rain", "class": "bus", "nearest_gt_id": null, "nearest_gt_iou": null, "equivalent_input_fp": false, "ranked_before_tp": true, "new_high_ranking_fp": true}`

- `rain/val/GOPR0572/GOPR0572_frame_000825_rgb_anon.png`；image_id=242，GT/候选类别=bus，分析方法=C1，GT id=None。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|

证据：`{"image_id": 242, "category_id": 5, "confidence": 0.35917118191719055, "box": "coordinates omitted from source snapshot", "matched_gt_id": null, "file_name": "rain/val/GOPR0572/GOPR0572_frame_000825_rgb_anon.png", "weather": "rain", "class": "bus", "nearest_gt_id": null, "nearest_gt_iou": null, "equivalent_input_fp": false, "ranked_before_tp": true, "new_high_ranking_fp": true}`

- `rain/val/GOPR0572/GOPR0572_frame_000825_rgb_anon.png`；image_id=242，GT/候选类别=bus，分析方法=guard，GT id=None。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|

证据：`{"image_id": 242, "category_id": 5, "confidence": 0.42412498593330383, "box": "coordinates omitted from source snapshot", "matched_gt_id": null, "file_name": "rain/val/GOPR0572/GOPR0572_frame_000825_rgb_anon.png", "weather": "rain", "class": "bus", "nearest_gt_id": null, "nearest_gt_iou": null, "equivalent_input_fp": false, "ranked_before_tp": true, "new_high_ranking_fp": true}`

## 现有证据无法区分

最多展示 3 个真实样例，优先雨天 bus。可用样例 3 个。

- `rain/val/GP030400/GP030400_frame_000129_rgb_anon.png`；image_id=206，GT/候选类别=bus，分析方法=C0，GT id=1202。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.0014 / IoU 0.4910 | 无匹配 |
| v4 | 无匹配 | 无匹配 | 无匹配 |
| guard | 无匹配 | bus / 0.0012 / IoU 0.5232 | 无匹配 |
| C0 | 无匹配 | 无匹配 | 无匹配 |
| C1 | 无匹配 | 无匹配 | 无匹配 |

证据：`{"input_hit": false, "candidate_hit": false}`

- `rain/val/GP030400/GP030400_frame_000129_rgb_anon.png`；image_id=206，GT/候选类别=bus，分析方法=C1，GT id=1202。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.0014 / IoU 0.4910 | 无匹配 |
| v4 | 无匹配 | 无匹配 | 无匹配 |
| guard | 无匹配 | bus / 0.0012 / IoU 0.5232 | 无匹配 |
| C0 | 无匹配 | 无匹配 | 无匹配 |
| C1 | 无匹配 | 无匹配 | 无匹配 |

证据：`{"input_hit": false, "candidate_hit": false}`

- `rain/val/GP030400/GP030400_frame_000129_rgb_anon.png`；image_id=206，GT/候选类别=bus，分析方法=guard，GT id=1202。

| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |
|---|---|---|---|
| input | 无匹配 | bus / 0.0014 / IoU 0.4910 | 无匹配 |
| v4 | 无匹配 | 无匹配 | 无匹配 |
| guard | 无匹配 | bus / 0.0012 / IoU 0.5232 | 无匹配 |
| C0 | 无匹配 | 无匹配 | 无匹配 |
| C1 | 无匹配 | 无匹配 | 无匹配 |

证据：`{"input_hit": false, "candidate_hit": false}`

## 可复现性与限制

`failure_cases.json` 保存全部 bus/train GT 对照、所有分类标签和原始证据、官方匹配的 FP、输入文件 SHA256。统计支持应读取主流程配对 bootstrap 报告；本脚本不将单例恢复解释为显著总体提升。

输入中的原始框坐标不用于训练；失配未解决时保留 unresolved。NMS 与定位边界案例应对照坐标和原图复核。
