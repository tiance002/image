# SafeRestore-RF：bus/train监督覆盖与丢检诊断（v2：原生候选几何勘误）

执行起点：`np2r-revision-20261026` / `fa209be9036a9619d9298535470b51574bccd73a`。诊断结束HEAD和分支不变，Git发布副本干净。真实科研区为`<local research workspace>`；`<public Git source snapshot>`仅用于核对版本，没有作为权重或数据运行区。

**根因状态：`BLOCKED_ROOT_CAUSE_UNRESOLVED`。已验证固定训练样本的监督证据链，也复核了全部验证集bus/train目标；现有证据尚不足以把总体损伤归为一种机制。本阶段训练步数、优化器更新、新检测预测、下载、安装、推送和部署均为0。用户另行明确允许原64张训练样本上必要的诊断前向/反向。**

## 1. 版本、输入和环境

Python：`<local yolo_daily Python>`，版本`3.11.15`；环境版本：`{'torch': '2.6.0+cu124', 'numpy': '2.4.4', 'ultralytics': '8.4.104', 'opencv-python': '5.0.0.93', 'Pillow': '12.2.0', 'PyYAML': '6.0.3'}`。每条命令设置`PYTHONPATH=third_party/pylibs;src;.`；原64张图像SHA另存`sample_image_hashes.json`（诊断运行期间采集，结束复查），未读取验证集像素。相关科研区源码逐项与HEAD相同，见`preflight.json`。

已完整读取指定的REVISION_REPORT、LEGACY_RESULT_STATUS、FINAL_DECISION与ACCEPTANCE_CRITERIA；以NP2-R修正结果及D0协议勘误为准。旧FINAL_DECISION中外部资源BLOCKED等时态已由NP2-R更新，本阶段不重新执行。原7项监督/12项恒等融合检查沿用，不重验颜色、融合、画质或外部模型。

原64张manifest逐项相同。训练与验证的image_id/GT_id属于不同命名空间；仅数值相同不形成对应关系。评估GT_id始终为source_annotation_id+1，CSV并列保存两者。以下“检出”均为IoU≥.5官方匹配。AP候选采用conf=.001/maxDets100，运行时采用独立conf=.25/maxDets300；NMS均类内IoU=.7/全局maxDet300，保留crowd规则。

## 2. 任务A：原生监督覆盖、损失和参数梯度

本诊断沿原训练式路径：RGB uint8 letterbox到640（含114灰边）后送复原，再检测；使用固定顺序、无随机翻转，不改旧固定样本的处理协议。当前冻结检查点上算出的任务梯度不是历史实际优化轨迹：C0历史未把GT梯度加入优化，v4也不能因此被称为接受过C1式GT监督；零正结果同样不能证明某GT整个训练期从未获正样本。64是唯一图像数；431是七类非crowd唯一GT数；每个模型各16个历史统计批（batch4）与64个补充梯度批（batch1）。三个模型重复观察同一数据，不能写成192张独立图、1293个独立GT，也不能把batch累计次数当作训练期曝光量。抽样组bus24/train24/other16不等于含类图数：实际bus出现在27张、train出现在25张。

| 模型 | 类别 | 含GT图数 | 唯一GT | 原生正anchor | 零正GT | 零正比例 | target_scores总权重 | 正anchor被忽略掩码覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | person | 29 | 81 | 397 | 9 | 0.111111 | 153.821937 | 13 |
| v4 | bicycle | 13 | 18 | 82 | 1 | 0.055556 | 27.152050 | 16 |
| v4 | car | 63 | 255 | 1456 | 9 | 0.035294 | 702.564158 | 57 |
| v4 | motorcycle | 8 | 10 | 48 | 0 | 0.000000 | 13.354903 | 13 |
| v4 | bus | 27 | 30 | 198 | 2 | 0.066667 | 75.915454 | 6 |
| v4 | train | 25 | 26 | 215 | 0 | 0.000000 | 74.436184 | 29 |
| v4 | truck | 9 | 11 | 74 | 0 | 0.000000 | 36.618314 | 0 |
| C0 | person | 29 | 81 | 392 | 9 | 0.111111 | 152.855206 | 13 |
| C0 | bicycle | 13 | 18 | 84 | 1 | 0.055556 | 27.349946 | 16 |
| C0 | car | 63 | 255 | 1460 | 9 | 0.035294 | 694.727824 | 54 |
| C0 | motorcycle | 8 | 10 | 48 | 0 | 0.000000 | 13.964148 | 14 |
| C0 | bus | 27 | 30 | 197 | 2 | 0.066667 | 75.891374 | 6 |
| C0 | train | 25 | 26 | 216 | 0 | 0.000000 | 75.454219 | 27 |
| C0 | truck | 9 | 11 | 75 | 0 | 0.000000 | 37.152785 | 0 |
| C1 | person | 29 | 81 | 398 | 8 | 0.098765 | 157.500753 | 13 |
| C1 | bicycle | 13 | 18 | 82 | 1 | 0.055556 | 27.292484 | 16 |
| C1 | car | 63 | 255 | 1455 | 9 | 0.035294 | 717.599759 | 53 |
| C1 | motorcycle | 8 | 10 | 49 | 0 | 0.000000 | 13.493834 | 15 |
| C1 | bus | 27 | 30 | 198 | 2 | 0.066667 | 77.067100 | 4 |
| C1 | train | 25 | 26 | 216 | 0 | 0.000000 | 78.107323 | 29 |
| C1 | truck | 9 | 11 | 75 | 0 | 0.000000 | 38.738333 | 0 |

以上正anchor逐GT与上一轮固定64统计一致。bus/train有正anchor的GT，其总target_scores均大于0；完整每个正anchor的索引、stride、中心和软分数已捕获于`native_positive_anchor_trace.csv`及各批JSON。分配器通过hook只读记录，每个诊断前向原生TaskAlignedAssigner只调用一次，保留全部GT联合分配；没有删类或重新分配。

1600张训练标签总体中bus有168GT/138图、train有212GT/184图；诊断覆盖其中30/26GT，分别约17.86%/12.26%。未观测其余1536张的原生分配，也没有历史整个优化期的分配/梯度轨迹。固定样本中的覆盖不能证明整个训练期监督充分。

### 2.1 损失：相同gain、坐标与归一化

下表复用已修正的16个batch4标量之和，分类正/负、box、DFL均乘原生gain（cls=.5、box=7.5、DFL=1.5）并除以同批target_scores_sum。它不是64图均值。目标框/anchor均按正确stride统一到网格坐标，权重保持[N,1]。新batch1逐项损失另存CSV，因tss/批大小不同，不直接把两种累计量相减。

| 模型 | 类别 | 分类正项 | 分类负项 | box | DFL |
| --- | --- | --- | --- | --- | --- |
| v4 | person | 2.233609 | 0.749515 | 3.893453 | 2.275659 |
| v4 | bicycle | 0.710653 | 0.044898 | 0.779714 | 0.426241 |
| v4 | car | 8.535725 | 1.546176 | 13.904928 | 10.188033 |
| v4 | motorcycle | 0.464479 | 0.042831 | 0.723073 | 0.369478 |
| v4 | bus | 2.276795 | 0.329669 | 2.172226 | 1.387755 |
| v4 | train | 2.184747 | 0.120987 | 2.097591 | 1.622716 |
| v4 | truck | 0.640563 | 1.011068 | 0.630746 | 0.605709 |
| C0 | person | 2.235877 | 0.800717 | 3.790413 | 2.235420 |
| C0 | bicycle | 0.714287 | 0.050045 | 0.814989 | 0.439844 |
| C0 | car | 8.476036 | 1.551123 | 13.914638 | 10.174937 |
| C0 | motorcycle | 0.468055 | 0.044175 | 0.743948 | 0.376442 |
| C0 | bus | 2.272678 | 0.306967 | 2.116179 | 1.372874 |
| C0 | train | 2.197740 | 0.149858 | 2.117163 | 1.650568 |
| C0 | truck | 0.651781 | 1.010939 | 0.636719 | 0.614965 |
| C1 | person | 2.144077 | 0.710373 | 3.702982 | 2.201548 |
| C1 | bicycle | 0.690054 | 0.044224 | 0.759427 | 0.416704 |
| C1 | car | 7.975004 | 1.518067 | 13.473626 | 10.054663 |
| C1 | motorcycle | 0.433963 | 0.035422 | 0.699445 | 0.367155 |
| C1 | bus | 2.199430 | 0.310528 | 2.046243 | 1.341045 |
| C1 | train | 2.246253 | 0.117298 | 2.172727 | 1.687775 |
| C1 | truck | 0.654385 | 0.921631 | 0.639034 | 0.638105 |

### 2.2 各项对复原网络的参数梯度

下表是64个单图批次中各项完整参数梯度向量L2范数的均值，使用FP64归约统计FP32梯度。范数不可加和，不能把四列相加称为类总梯度；也不是整个数据集梯度的范数。“分类正项”按target_scores>0的匹配类别位置分组，并不等于梯度一定提升类别分数；软目标BCE的logit导数为sigmoid(logit)−target，可有不同符号，本轮未保存每个anchor的全部logit来宣称正项都具有有益方向。模型前向沿用历史精度，反向cuDNN TF32关闭，未放宽1e-4校验阈值。这里是原生任务分支梯度，未乘历史C1外部GT系数0.009570541647407782及其warmup，也未与A流复原损失相加；不能将这些范数解释成历史总优化梯度的贡献率。

| 模型 | 类别 | 分类正梯度范数均值 | 分类负梯度范数均值 | box梯度范数均值 | DFL梯度范数均值 | 四项向量和的范数均值 |
| --- | --- | --- | --- | --- | --- | --- |
| v4 | person | 2.082257 | 2.507980 | 2.534334 | 0.698484 | 5.909571 |
| v4 | bicycle | 0.631017 | 0.109225 | 0.967848 | 0.187896 | 1.403456 |
| v4 | car | 5.661875 | 6.222344 | 11.729476 | 2.416512 | 18.909755 |
| v4 | motorcycle | 0.266057 | 0.134590 | 0.273141 | 0.101385 | 0.593809 |
| v4 | bus | 2.066980 | 0.891383 | 1.992913 | 0.640652 | 4.507991 |
| v4 | train | 2.294857 | 0.596103 | 2.893738 | 0.803981 | 5.213930 |
| v4 | truck | 0.315110 | 2.545086 | 0.442937 | 0.068336 | 3.099210 |
| C0 | person | 2.122091 | 2.985925 | 2.706362 | 0.652017 | 6.362724 |
| C0 | bicycle | 0.708413 | 0.128151 | 0.791114 | 0.224928 | 1.318569 |
| C0 | car | 5.776461 | 6.278276 | 11.658918 | 2.451428 | 19.022971 |
| C0 | motorcycle | 0.259721 | 0.160003 | 0.266766 | 0.102743 | 0.612274 |
| C0 | bus | 1.937009 | 0.799885 | 1.995956 | 0.646802 | 4.426581 |
| C0 | train | 1.948159 | 0.698536 | 2.606640 | 0.664469 | 4.435519 |
| C0 | truck | 0.282363 | 2.319705 | 0.513877 | 0.080243 | 2.934339 |
| C1 | person | 2.072025 | 2.296990 | 2.491601 | 0.654434 | 5.488875 |
| C1 | bicycle | 0.656404 | 0.113001 | 0.829096 | 0.268470 | 1.330670 |
| C1 | car | 5.044473 | 5.453676 | 10.990287 | 2.306452 | 17.084618 |
| C1 | motorcycle | 0.260038 | 0.125226 | 0.281411 | 0.090119 | 0.568103 |
| C1 | bus | 1.949529 | 0.738443 | 1.999551 | 0.612935 | 4.262746 |
| C1 | train | 2.053481 | 0.517029 | 2.809165 | 0.676857 | 4.666484 |
| C1 | truck | 0.301266 | 2.028524 | 0.456411 | 0.066569 | 2.554722 |

未对单个GT隔离反传：逐GT记录的是联合分配后的anchor与权重，梯度是类/图内全部GT的联合归因，不能当作每个GT的独立梯度。逐图、逐类、逐项损失/梯度/与原生向量点积均在`component_gradients_per_image.csv`。每批将28个实际分量梯度向量相加，并与原生总量比较；同时分别将各类分类、box、DFL向量和与对应原生分量比较。类总向量由四项向量相加定义；有效FP64早期批还另算过类总反向，后续避免重复这7次反向。主检验始终是独立的28项/原生总量及3项原生分量向量比较，不仅比较范数。

### 2.3 零正样本和忽略掩码

v1报告的“无原框内中心导致原生候选缺口”已撤回。两个零正bus均位于同一张`night/train/GP010376/GP010376_frame_000009_rgb_anon.png`（train image_id766）：source5123/eval5124有一个stride8中心(316,324)在框内，source5124/eval5125在stride8/16/32均无框内中心。旧“candidate_centers_in_ignore”仅统计原框严格内部中心与BCE忽略掩码的交集，不能当作原生扩展候选的忽略统计。本地固定Ultralytics8.4.104的select_candidates_in_gts会把小于stride[0]=8的宽或高替换为stride_val=stride[1]=16后筛候选，而回归仍针对原GT。用该原生方法对完整GT几何离线重放，两个bus各有stride8的4个、stride16的1个、stride32的0个候选（各5个），最终正anchor却都是0；二者均不能归因于没有原生几何候选。没有保存对齐分数/top-k/冲突阶段张量，具体分配损失机制仍UNRESOLVED。

| 模型 | 分类正loss | 分类负loss | box loss | DFL loss | 分类正grad | 分类负grad | box grad | DFL grad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | 0.000000 | 0.004996 | 0.000000 | 0.000000 | 0.000000 | 0.132064 | 0.000000 | 0.000000 |
| C0 | 0.000000 | 0.005589 | 0.000000 | 0.000000 | 0.000000 | 0.198320 | 0.000000 | 0.000000 |
| C1 | 0.000000 | 0.004516 | 0.000000 | 0.000000 | 0.000000 | 0.130069 | 0.000000 | 0.000000 |

这张图确实输入了两个bus标签，但其bus分类正项、box、DFL监督均为零；类总梯度仍可因负分类BCE非零。由此撤回任何用“整体/类别任务梯度非零”替代正监督覆盖的推断。全样本仍有28个bus GT和全部26个train GT获得正anchor，不能从该单帧反推整个类别失效。候选中心最终归属中，第一个零正bus的5/5候选被其他GT占用，第二个为3/5（另2个未分配）；这只能观察最终占用，不能证明发生了冲突阶段抑制，因为bus可能在top-k之前已经没有有效候选。

固定64中忽略区域共19个：rider 7、crowd 12、额外ignore 0。bus/train没有因这些官方标记被排除的GT。规则`(~mask) | (target_scores>0)`保留正软标签，只去掉相应负项；73个未标注COCO类别的class_weights为0，7个真实监督类为1，框/DFL不应用该负项掩码。5项张量检查验证正软标签及其梯度被保留（含1e-12软目标），忽略负项梯度为零。未对官方crowd语义重新人工标注，因此不声称标签本身绝对无误。

## 3. 任务B：全部真实bus/train验证案例

全部406张既有input/v4/C0/C1预测经官方COCO重新离线匹配，95个非crowd bus/train GT来自86张唯一图像，按image_id+GT_id去重，主矩阵95行，每行四方法并排；380行方法观察表是95×4，不能称为380个目标。全部命中状态与旧正式预测/修订雨雪矩阵一致；仅复算匹配，没有生成新模型预测。

| 范围 | 类别 | GT | input .25/.001 | v4 .25/.001 | C0 .25/.001 | C1 .25/.001 |
| --- | --- | --- | --- | --- | --- | --- |
| all406 | bus | 49 | 9 / 15 | 6 / 12 | 6 / 11 | 6 / 12 |
| all406 | train | 46 | 18 / 20 | 17 / 19 | 16 / 21 | 16 / 20 |
| fog | bus | 5 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| fog | train | 3 | 2 / 2 | 2 / 2 | 1 / 2 | 1 / 2 |
| rain | bus | 32 | 5 / 9 | 3 / 7 | 3 / 7 | 3 / 7 |
| rain | train | 9 | 2 / 3 | 2 / 3 | 2 / 3 | 2 / 3 |
| night | bus | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| night | train | 24 | 7 / 8 | 6 / 7 | 6 / 9 | 6 / 8 |
| snow | bus | 12 | 1 / 3 | 0 / 2 | 0 / 1 | 0 / 2 |
| snow | train | 10 | 7 / 7 | 7 / 7 | 7 / 7 | 7 / 7 |

表中.001只指该协议IoU=.5的GT命中数，不是AP。夜天bus GT=0，AP为N/A，不记0分。相对input，三模型bus均丢失GT1269/1673/2623且无新增；v4 train丢失852，C0/C1丢失23/931，均无相对input新增。C0和C1的运行时bus/train检出集合完全相同。C0的train低阈值命中21个虽多于input20个，运行时16个仍少于input18个；候选命中数量不能代替AP排序或运行时指标。

### 3.1 逐类AP复核（conf=.001官方COCO）

| 方法 | mAP50 | mAP50:95 | 6类不含train | person | bicycle | car | motorcycle | bus | train | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| input | 0.233510 | 0.138682 | 0.208627 | 0.260138 | 0.081542 | 0.557207 | 0.072163 | 0.121324 | 0.382810 | 0.159388 |
| v4 | 0.220924 | 0.127568 | 0.198921 | 0.269798 | 0.067000 | 0.555463 | 0.063184 | 0.088846 | 0.352937 | 0.149238 |
| C0 | 0.214648 | 0.126094 | 0.198148 | 0.265835 | 0.061383 | 0.556141 | 0.070210 | 0.097160 | 0.313652 | 0.138155 |
| C1 | 0.215576 | 0.127859 | 0.199590 | 0.269461 | 0.055107 | 0.563299 | 0.070472 | 0.099366 | 0.311490 | 0.139836 |

各类GT数：person=518, bicycle=68, car=1836, motorcycle=59, bus=49, train=46, truck=106。遵从D0新协议勘误，不复述历史截断AP中motorcycle零分的能力上限解释。

### 3.2 关键GT证据（运行时.25；括号内为.001同类IoU≥.5最高分候选）

| val image_id | eval GT_id | 类别/天气 | 原图W×H | input | v4 | C0 | C1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 23 | train/fog | 718×634 | hit (0.3095 / 0.9316) | hit (0.3372 / 0.8982) | miss (0.0232 / 0.7701) | miss (0.0240 / 0.7740) |
| 137 | 852 | train/night | 764×166 | hit (0.4248 / 0.9516) | miss (无同类IoU≥.5候选) | hit (0.4643 / 0.9744) | hit (0.4711 / 0.9746) |
| 153 | 931 | train/night | 720×724 | hit (0.6208 / 0.6276) | hit (0.8257 / 0.5217) | miss (0.0044 / 0.5794) | miss (0.0045 / 0.5792) |
| 214 | 1269 | bus/rain | 397×126 | hit (0.3678 / 0.5449) | miss (0.1587 / 0.5446) | miss (0.2139 / 0.5426) | miss (0.1994 / 0.5414) |
| 271 | 1673 | bus/rain | 915×1080 | hit (0.5870 / 0.9619) | miss (0.0168 / 0.8595) | miss (0.0596 / 0.9137) | miss (0.0732 / 0.9279) |
| 383 | 2623 | bus/snow | 327×121 | hit (0.2675 / 0.6229) | miss (无同类IoU≥.5候选) | miss (无同类IoU≥.5候选) | miss (0.1901 / 0.6258) |
| 395 | 2730 | bus/snow | 163×40 | miss (无同类IoU≥.5候选) | miss (0.0066 / 0.5032) | miss (无同类IoU≥.5候选) | miss (无同类IoU≥.5候选) |
| 264 | 1626 | bus/rain | 82×185 | miss (无同类IoU≥.5候选) | miss (无同类IoU≥.5候选) | miss (无同类IoU≥.5候选) | miss (无同类IoU≥.5候选) |

- **直接观察**：雨天GT1269、1673均有低阈值正确重叠框，但复原后分数落到.25以下；原尺寸分别397×126和915×1080，不能把它们统称为小目标。该观察不解释分数下降的内部原因，也不认为不同方法的最高分框具有相同anchor身份。
- **类别竞争候选**：night train GT852在v4中存在bus框，score=.587626、IoU=.950069；input已同时存在bus=.386522的重叠候选。C0/C1仍检出train。可验证重叠类别错误，无法从仅最大类别候选恢复全部80类logits或断言唯一机制。
- **定位候选**：GT1626在input原本就漏检；input/C0/C1高分bus框IoU约.4534/.4654/.4663，均未到.5。这是输入已有定位短板，不算复原新增丢失。
- **NMS未解决**：GT931的C0/C1最高分同类框仍很高（.802718/.802435），但IoU=.490166/.497059；另有pre-NMS候选IoU=.500192/.501601且与更高分框强重叠。缺少原生anchor/NMS索引，统一标`UNRESOLVED_NATIVE_NMS_TRACE`。对候选平移±2像素的离线敏感性范围可跨过.5，仅说明边界敏感性，不宣称原坐标实际有2像素误差。不能把旧“满足几何条件的NMS证据”升级成已追踪算子抑制原因。input中24个GT存在类似decoded抑制候选对而仍hit，说明候选对本身不等于丢检原因。
- **协议限定**：GT2730/source2729在v4的.001协议hit，在所有方法.25协议miss；输入同样miss。不得把它写成运行时恢复。

## 4. 训练监督与失败案例之间能建立什么联系

以下是同一训练帧、同一GT和同一检查点上的可复核连接，复用原24张尺度诊断的D/A/B检出结果，没有再推理。A先缩放再复原，B原分辨率复原再缩放；共同padding/FP32几何已修正，但上下文/分块仍同时变化。监督前向沿历史TF32并在letterbox灰边后复原；尺度主对照TF32关闭，A在未pad的缩放图上复原后共同padding，B在原分辨率复原后共同padding。因此训练式监督的anchor/梯度不等于主路径A本身的分配；这是同GT跨诊断关联，不冒充同一处理/数值计算图上的因果干预。

| 模型 | train image_id | eval GT_id | 类别 | 正anchor | target权重 | 同图类总梯度范数 | D .25 | A .25 | B .25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | 367 | 2067 | bus | 10 | 6.251917 | 5.769229 | hit | miss | hit |
| v4 | 1398 | 8750 | bus | 10 | 5.126446 | 11.040849 | hit | miss | hit |
| C0 | 1109 | 6873 | train | 10 | 6.197934 | 7.354794 | miss | hit | miss |
| C1 | 1109 | 6873 | train | 10 | 6.078463 | 9.995498 | miss | hit | miss |

类总梯度范数包含该图该类负项，不是该单个GT的梯度。上述三种GT在训练式监督诊断中都已有10个正anchor，却仍可在另一路径丢检；这排除了“这些特定目标完全未输入正监督”的描述，未证明损伤由某一梯度项或tile造成。

固定64训练帧与95个val GT所在图没有相同路径。75/95个val GT的天气+原始sequence键在固定64中出现；这只是相关序列暴露，不能证明是相同物体、相同anchor或某次历史优化梯度造成丢检。完整链接表只按真实路径/序列建立，不用数值ID跨split硬连。未观测历史全训练期，因此GT→最终参数变化→val损伤的因果链仍缺证据。

## 5. 检查结果、未执行项及资源记录

| 模型 | 有效单图梯度批 | 原生分量最大标量误差 | 向量最大相对L2 | 向量余弦最小 | 向量最大绝对误差 | 对旧原生loss最大差 |
| --- | --- | --- | --- | --- | --- | --- |
| v4 | 64 | 0.0000009537 | 2.156e-06 | 0.999999999998 | 2.813e-05 | 0.0000000000 |
| C0 | 64 | 0.0000004768 | 2.161e-06 | 0.999999999998 | 2.289e-05 | 0.0000000000 |
| C1 | 64 | 0.0000004768 | 1.988e-06 | 0.999999999998 | 2.241e-05 | 0.0000000000 |

每批4组独立向量比较，总计768组（192批×总量/分类/box/DFL）；均通过1e-4相对L2阈值。此前48个batch4原生损失校验及192个逐类总梯度校验也逐条离线复核。新增5项mask检查、4项17,061,219维长向量FP64统计检查均通过。

原生监督坐标转换、[N,1]形状和固定gain沿用修正版；新梯度脚本没有修改训练loss、分配器或模型架构。逐图全部GT在一个真实原生assigner调用中分配，hook仅记录。模型参数、检测器BN、3个checkpoint和所有37项输入哈希前后不变。整个阶段没有构造优化器，autograd.grad不向parameter.grad累积更新。

执行中发现并保留了以下诊断资源/统计版本（仅涉及本阶段新增诊断；旧NP2-R实验不覆盖）：

- 初始参数向量驻留GPU导致显存压力；改为CPU累加，并显式释放最后循环BCE标量持有的图、每批清理缓存。有效完成批次文件未被覆盖。
- CPU FP32长向量dot/norm出现余弦>1的数值错误（例如1.003058）。21个初期完成记录及中断日志保留为“旧版本待修订”，不用于最终表。最终数据在`fp64_component_gradients/`重新完成，模型/梯度仍FP32，统计点积及范数平方改FP64 einsum并通过独立NumPy double参考。
- 最终FP64首批中的11个有效记录以SHA保留并续跑；只删除冗余的7次类总反向与空集合零导数反向，28项及原生3分量的独立向量检验保持。部分源文件继承了“old completed batch0 retained”描述；它仅属于先前资源续跑说明，最终目录绝未复用FP32余弦缺陷版记录，使用范围以`PARTIAL_GRADIENT_METRIC_STATUS.json`、`fp64_minimal_resume.json`和最终目录为准。
- 完整有效诊断为192次训练图前向；另有21次旧统计版已完成批及若干被中断未完成批，均来自同一固定64张。它们不算新增独立样本或完整成功实验。没有val反向/前向、没有NMS预测生成，没有全量图像推理。

最终有效诊断计数：`{'restoration_forward': 192, 'detector_forward': 192, 'native_assigner_calls': 192, 'autograd_calls': 3784, 'optimizer_created': 0, 'optimizer_steps': 0, 'prediction_generation': 0, 'validation_forwards': 0}`。中断批未保存完整计数，不能把成功计数192说成整个阶段所有尝试的总前向数。保留两种执行源SHA与续跑哈希；不把不同资源尝试当作多随机种子实验。

新增最终核验曾错误断言“正anchor必须在原GT框内”。607个点超出原框1e-4容差，另有边界点；合计650个不满足原框严格内部。核对固定源代码发现原生小框候选扩展，原生方法重放确认7415/7415实际正anchor都在正确候选集合内。原断言失败与v1报告保留并标旧版本待修订，当前v2撤回几何根因；并非修改分配器或放宽原生损失/梯度阈值。资源齐备，没有下载/安装阻断。证据阻断包括：全训练期原生分配/优化轨迹缺失；两个零正bus的TAL中间打分/top-k/冲突trace均缺失；val原生NMS索引、所有anchor全80类logits和未裁剪decoded坐标缺失。这些均记UNRESOLVED，未通过新全量推理补齐。Histoformer、AdaIR、颜色审计、画质、C2、任何训练和部署均未执行。

## 6. 根因判断、置信等级与唯一候选实验

| 判断 | 等级 | 依据/边界 |
| --- | --- | --- |
| 固定64中多数bus/train获得非零正权重；类/图汇总可见正项/box/DFL参数梯度 | 已验证事实，高 | 正anchor覆盖28/30bus、26/26train；没有逐GT独立反传，不宣称每个GT独立梯度已测 |
| 同一帧两个bus各有5个原生几何候选，却均无最终正监督 | 已验证局部缺口，高 | 原生方法几何重放+真实最终分配；不能据原框中心数归因，也不外推全类 |
| crowd/rider负项掩码删除了这些已分配bus/train正监督 | 无支持，拒绝该局部解释 | 正软目标保留；忽略掩码不作为TAL输入，且保留已分配正软标签 |
| 雨天两bus发生低阈值正确框分数跨.25阈值下降 | 已验证观察，高 | 同协议保存预测；为何分数下降仍未知 |
| 总体损伤主要由小目标、NMS或整体监督不足造成 | UNRESOLVED | 大小目标均有丢失、NMS缺trace、未观测训练期 |
| 处理路径/分块上下文是值得检验的机制 | 待验证假设，中低 | 既有同GT命中变化有证据，但A/B仍含上下文/分块混杂 |

**当前结论为`BLOCKED_ROOT_CAUSE_UNRESOLVED`：不推荐直接增加总体检测损失权重，不启动训练。**

唯一下一步：**固定C1及原64张的原生TAL阶段只读追踪**；当前不推荐模型训练效果实验。本阶段只提出，不执行。v1中的tile640/1024候选保留作历史方案，最终优先级由本段覆盖。

| 设计项 | 预先固定内容 |
| --- | --- |
| 唯一变量 | 记录粒度：仅最终分配→只读记录原生中间阶段；模型/算法行为完全不变。这是诊断追踪实验，不冒充训练效果消融 |
| 固定检查点 | runs/gt_next_phase_20261025_C1/latest.pt / e17b1940179e00b705ee3f5bd5e828555de0b6ca4c86d043af7bbe1d288d91ff |
| 数据与顺序 | 原64张train manifest及原顺序，全部GT联合分配，不删类/不挑有利图 |
| 训练预算/优化器 | 两组均0步，无优化器；现有C1原生64记录作观测对照 |
| 其他固定项 | 同640letterbox灰边复原路径、batch1、seed42无增强、forward TF32/backward FP32、冻结YOLO、nativeTAL参数/gain/masks；若核对预测仅复用现有.001/.25协议 |
| 需捕捉 | 扩展候选mask、每GT候选分数/预测框/IoU/align metric、top-k、冲突前mask_pos、最高重叠消解前后和最终GTidx/target_scores |
| 支持数据 | 两个零正bus均有5候选而最终0正；其他GT最终占用5/5及3/5，但缺阶段trace，无法区分打分/top-k/冲突 |
| 可能改善 | 降低零正GT机制的UNRESOLVED比例；mAP/Recall不期望改变，因为被观测算法必须保持不变 |
| 潜在副作用 | 记录增加显存、I/O、延迟；若改变原生结果则追踪无效 |
| 停止条件 | 检查点/样本/精度/原生损失/anchor索引/target_scores不一致，额外调用assigner，BN/参数或优化器变化即停止；OOM/缺字段记INCOMPLETE，不换算法/精度/删GT |
| 拒绝采纳条件 | 不能复原原生最终分配或仍分不清阶段，则继续BLOCKED；单帧机制不推出整个训练期或验证集根因，不据此直接改权重或分配器 |

若后续真的选择任何处理路径训练，应另行审查并保证同初始化、相同实际启用的GT任务梯度、数据/优化预算和评测，区分裁剪采样；本阶段既未提出新的4000步配方，也未执行。

406/200均为已用于历史模型选择的validation；训练诊断资源同样参与过历史训练，帧与序列相关、单seed/单检查点及少量bus/train限制外推。重新拆分这些帧不能制造独立测试。本报告不宣告完整D1/D2通过。

## 7. 交付索引与输入哈希

- `preflight.json` / `preservation_checks.json`：版本、环境、37项输入前后哈希。
- `fixed64_manifest.json` / `training_population_counts.json`：唯一图像与标签总体口径。
- `supervision_per_gt.csv`、`supervision_per_image.csv`、`supervision_per_batch4.csv`：旧有效覆盖及原始64逐图/逐批链接。
- `component_gradients_per_image.csv`、`component_native_batches.csv`、`component_vector_checks.csv`、`component_loss_gradient_summary.json`：新逐项loss/参数梯度与每批向量校验。
- `native_positive_anchor_trace.csv`及`fp64_component_gradients/component_batches_*/*.json`：真实联合分配的GT→anchor索引/中心/stride/软分数、每项梯度范数及原生向量比较。
- `bus_train_gt_matrix_all406.csv`（95行）、`validation_cases.json`、`validation_case_observations.csv`：四方法并排状态及逐例分数/定位/类别观察。
- `same_train_gt_evidence_links.json` / `validation_training_links.json`：相同训练GT直接关联与跨split可验证序列关系。
- `nms_evidence_unresolved.json`：decoded候选对，不能当原生算子trace。
- `native_candidate_geometry_audit.json`、`PINNED_NATIVE_TAL_SOURCE.txt`、`preflight_native_source_addendum.json`：实际固定版本的小框扩展规则和原生几何重放；7415个实际正anchor均在其候选集合内，650个不满足原框严格内部条件（含边界），不能据此判为坐标错误。
- `zero_bus_candidate_final_occupancy.json`：候选最终占用，不能代替冲突阶段trace。
- `mask_contract_checks.json` / `fp64_metric_checks.json` / `offline_checks.json`：检查结果。
- `next_minimal_experiment_v2.json`：最终唯一下一步设计与停止/拒绝条件；v1候选仅存档。
- `offline_audit.py` / `component_gradient_probe_fp64_minimal.py` / `assemble_report.py`：复核源码；均使用新目录独占写入，拒绝覆盖已有结果。

输入SHA256：

| 输入文件 | SHA256 |
| --- | --- |
| results/np2_revision/REVISION_REPORT.md | 0a3ff8d4845a2d60ad845078d0ad211d7d781b463221eb07b568ad8b0d09cc4d |
| results/np2_revision/LEGACY_RESULT_STATUS.md | 51758803505afa37197f73a0bdeb68482b7a6ae1cb2f2e3438313239bb935874 |
| results/next_phase/FINAL_DECISION.md | 8e0072524b582bc2c8d251233c09ebc9c1278acebcc3782a1306a09e6d23b7c6 |
| docs/ACCEPTANCE_CRITERIA.md | 09e605187104fcc99d9501657411ca1631aac3386235324da1dfa752669562e8 |
| results/np2_revision/supervision_probe.json | 9ce1a0b69809784be7ce6c12873e309bdb7b3184a1220b9a7d3de369afbcc558 |
| results/np2_external_probe/supervision_probe.json | 0ec525f16b8fe1182f74580d8bf225e80403609820547aee6f6c8fc31112a8be |
| results/np2_revision/scale_order_probe.json | 50ee135b9ea6a643f9f4f2512a7c8fc363f45b16d3c5a067672cb0d782e42353 |
| results/next_phase/failure_cases.json | 9e2f1a169ac541696473a906096f3308de8e737050d60f16b325c266c38299b9 |
| results/next_phase/focus_changes.json | 10f608b749350bbfc9b25f4b90bf6b5edefabacb9a9c584dbc30e8d0fdc46e54 |
| results/next_phase/metrics.json | b37ec58e16b60e4d957939cbd860d4d1be372b9cc85f040df4a1c11d0f41e738 |
| results/np2_revision/bus_train_gt_matrix.csv | d1b468fdb00d1e8ff4581c5593f43381da1dfc1e429b0b5e79e436e34f51df21 |
| configs/next_phase_evaluation.yaml | b14a19cf70aa3d7d03cf36a56196f0f0a5317f6a9cde3dd73950139078718ebb |
| data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json | 33be9416b6c4c66d55fe61ee0d3dd1e2ecf06fc82db38480aa377e5341c24487 |
| data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json | 1dc7b900b8f1f2797f2c9238417d438d336ac8907778940402fab852ee434215 |
| src/saferestore/losses/task_detection.py | 810d614ac790df1507163bd905fee368ec454d90c1abd0e48ebb649216dd5b7f |
| src/saferestore/data/detection_gt.py | c1b865558334b00dbf509d72c92371fada3b5bec401ffbdddb46f85e180d43c7 |
| src/saferestore/metrics/next_phase.py | 746b7fe2c6f02daa569f7a5c8d07859371cd6f57c8890e8b6bb7c8ba1622ec4d |
| scripts/np2r_supervision_probe.py | 2d01e79f9420678b94b3197660bdf5b17d1c6b078faf0b90a34c287bba58ad80 |
| scripts/np2_scale_order_probe.py | 9e6f5929ee50d9e39a87fa3170dfdfb787fde008f6a025ff31c67c4ac0764bdc |
| scripts/evaluate_next_phase.py | 3d3fd42d312810ad9765edac26c08104a2bb72c54b953bb2780c53f0cff0b2d8 |
| scripts/analyze_next_phase_failures.py | a1e6e0bef406bf642e07476fef37760b77c272cff00302311e7d1d287baed2da |
| yolov8n.pt | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 |
| runs/real_v4_allweather/latest.pt | 8221a9037db0a907dc5d56698b209299061fccf0f01ac72b700cd57910d768d9 |
| runs/gt_next_phase_20261025_C0/latest.pt | 512b8677945ceee1161c568081d9660fcaaf2f1a3eb02b54f0be55405e6bd3ad |
| runs/gt_next_phase_20261025_C1/latest.pt | e17b1940179e00b705ee3f5bd5e828555de0b6ca4c86d043af7bbe1d288d91ff |
| results/next_phase/predictions/input/ap001.json | 8979526afbbe6f1d7c4c670fb5cbea76944f2875320b187303a59166b31cab55 |
| results/next_phase/predictions/input/runtime025.json | 482b58db4d2260722cf6722d67e8394e379de846e852520bc500a1f01554903d |
| results/next_phase/predictions/input/metadata.json | 213d97caa98067a4712f71b6f46a72f66821fd40528df643f2aa5dbf937981b8 |
| results/next_phase/predictions/v4/ap001.json | 99b6f0f61454c67b7c586ca0cec435c6b2fbf2b5c900149f897020e81f649234 |
| results/next_phase/predictions/v4/runtime025.json | 81ea0aba23855f4a6969091b7430a636b7e11cab2282eccfeb1f37b9f53d854b |
| results/next_phase/predictions/v4/metadata.json | cebfb01011b5fcb4616fb7fb7d8b3ff26d5970122e35055c448fc9b913433f61 |
| results/next_phase/predictions/C0/ap001.json | 932ff0fabf2683e1b7b9e005638d3cfdf3c4d54c31e1cde0a3a0a05e48ca6cbe |
| results/next_phase/predictions/C0/runtime025.json | f8b425c7fe39572a40fedb0dda260faf27b6346b21f6d453a7b59ad6c26a70f3 |
| results/next_phase/predictions/C0/metadata.json | 7bf6e2d09e1c8b37a8dcf209e32611043b6b03d5ee4b503a207910647f23d7f2 |
| results/next_phase/predictions/C1/ap001.json | 26d50a2abe645798d28866d44d192cce3c9d1c8b5f89103febcc4c404e0e7561 |
| results/next_phase/predictions/C1/runtime025.json | 4286f46b496c916b4004b0ceb994b976eed4b5005a04f71ec868d619e8af867c |
| results/next_phase/predictions/C1/metadata.json | 0aa12937d5e8106e8d2cff1aaa1f0b75a0e808849ba7cd245bd2fc6aade6702a |

完成后停止，等待负责人审查；不自动进行下一实验、提交/推送、merge、部署或替换默认模型。
