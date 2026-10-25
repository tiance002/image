# P0：RGB/BGR 实物核验与数据修复

## 已证实的事实

不能仅凭源码判定历史训练数据是否真的发生颜色交换。本次读取原始 ACDC PNG、已有训练 JPEG、CSV 清单与实验配置，执行了逐字节重编码核验，证据保存在 `color_audit.json`。

| 数据 | 历史配对数 | 唯一目标图 | 真实文件核验 |
|---|---:|---:|---|
| `data/real_scene/manifests/train.csv` | 800（night400、snow400） | 800 | **800/800 目标 JPEG 与交换 R/B 的重编码结果逐字节相同；0/800 与正确颜色结果相同** |
| `data/real_weather/manifests/train.csv` | 2400（fog1200、rain1200） | 800 | **800/800 清晰目标 JPEG 正确编码，未发生全图 R/B 交换** |

历史自监督数据先在原始 1920×1080 图上合成并编码 JPEG95，随后 `scripts/downscale_selfsup.py` 读回、缩到 960×540、再编码 JPEG95。本次核验复演了这一压缩链路，不能把 1920×1080 PNG 与 960×540 JPEG 直接比较。首个 night 退化输入 `GOPR0351_frame_000159__selfsupnight.jpg` 也与错误通道顺序的完整随机合成及双重 JPEG 链路逐字节相同（seed=20261025）。因此，夜雪自监督配对的输入与目标都存在交换问题，二者仍空间对齐，但模型实际看到的颜色分布错误。

具体例子：`snow/train/GOPR0122/GOPR0122_frame_000161_rgb_anon.png` 的历史目标 `GOPR0122_frame_000161__selfsupsnow.jpg`，RGB MAE 为 5.643/255，反转 JPEG 的 R/B 后为 1.512/255；精确编码匹配证据比这种近似 MAE 更强。夜间两例与逐文件 SHA256 也在 JSON 中。

## 雾雨标定问题与证据边界

历史 `data/real_weather/calibration.json` 的大气光 `air` 和雨天 `mean_c/std_c` 来自 OpenCV BGR 数组。重新读取本地真实 train PNG，按原统计过程计算的第一条 fog/rain 参数与文件数值相同。例如第一条 fog 大气光按 BGR 为 `[0.9408,0.9762,0.9888]`；RGB 合成必须使用逆序 `[0.9888,0.9762,0.9408]`。当前旧源码确实把这些 BGR 参数直接用于 RGB 图像，属于确定的生成器契约错误。

**但历史雾雨输入的配方溯源尚不完整。** 用当前源码、历史标定、默认 seed42 复演每种天气前三个历史输入，6/6 都未实现逐字节匹配；另检查若干常见 seed 也不能还原首个 fog 输入。不能把当前生成器的静态错误直接当成所有历史 2400 张雾雨退化输入都已被实物证实受同一错误影响。历史配方/seed 或生成时源码存在未记录差异，本轮明确保留这一技术债。

## 涉及版本

按各运行目录实际 `config.json`：v1/v2/v3 使用旧雾雨清单；v4/v5/v6/v7，以及上一轮 confidence control/guard 使用旧雾雨清单和已证实交换的夜雪清单。完整目录列表见 JSON。旧配置引用证明的是使用的路径；早期版本没有逐文件内容清单，不能由此恢复每次运行时所有文件的历史内容。上一轮 control/guard 冻结的两个 CSV 哈希与本次审计一致。

v4 初始化仍来自历史数据分布。后续 C0/C1 即使使用修复后的数据，也只能把两组之间差异归因于新增检测监督；不能把“修复数据后普通续训”的收益计入新增模块。

## 最小修复

- `scripts/build_real_scene_pairs.py`：合成维持 RGB，JPEG 编码前显式 RGB→BGR；保留原始分辨率下的退化、seed20261025、双重 JPEG95/INTER_AREA960×540 顺序。默认写入新目录，拒绝覆盖已有目录。
- `scripts/build_real_weather_pairs.py`：在合成入口把历史 BGR 标定的 `air/mean_c/std_c` 转为 RGB；原有清晰图和 JPEG 写入链路保持。新增输出路径参数、数据颜色契约侧车文件和禁止覆盖已有数据/标定表的检查。
- 保留所有旧 CSV、JPEG、标定、checkpoint 和历史报告。
- `scripts/audit_pair_color.py`：提供历史实物核验、雾雨输入溯源检查、新版数据完整性核验入口。
- `tests/test_pair_color_contract.py`：10 项专项测试通过，覆盖 RGB 编码与训练 RGB 加载闭环、缩图链路、fog/rain 标定顺序和禁止覆盖。

## 新版数据与复现

本次 CPU 生成，无下载或依赖安装：

- `data/real_weather_rgb_v2/manifests/train.csv`：2400 对，seed42，每清晰图3次合成。
- `data/real_weather_rgb_v2/manifests/dev.csv`：200 对，seed42，每清晰图1次合成。仅用于合成配对质量评估，**不得作为真实天气配对真值**。
- `data/real_scene_rgb_v2/manifests/train.csv`：800 对，seed20261025，night400/snow400。

各目录颜色契约 JSON 保存尺寸、seed、CSV 和标定 SHA256。完整生成后的逐文件解码、文件流 SHA256、目标重编码匹配与样本顺序检查记入 `color_audit.json` 的 `fixed_version_validation`。

最终完整核验已通过：5200 个新版唯一 JPEG 全部可解码，尺寸均为 960×540×3，坏图0；三份 CSV 的样本 ID、天气、sequence 及顺序与旧版相同。新版 scene 的800个目标全部与正确颜色原始 PNG 的指定压缩链路逐字节匹配；雾雨 train 的800个清晰目标、dev的200个清晰目标与旧版正确目标逐字节相同。旧版两份 train CSV 与历史标定表 SHA256 均未变化，并与上一轮 guard 冻结的 CSV 哈希一致。

| 新版划分 | 配对 | 唯一 JPEG | CSV SHA256 |
|---|---:|---:|---|
| scene/train | 800 | 1600 | `06524465ec0e51335be87d05e2e1b5e97361d1254a4084c6f1258a1243ce0e40` |
| weather/train | 2400 | 3200 | `7a531ea9bb211d9c1494bc7790935ce9edd303bf541ec01af79971fcad690bb7` |
| weather/dev | 200 | 400 | `e002e57820b8039cf4d1970107bd04821dbb7cebad6c5abef375e2010397206f` |

```powershell
python scripts/audit_pair_color.py
python scripts/audit_pair_color.py --replay-inputs
python scripts/build_real_weather_pairs.py --split train --per-image 3
python scripts/build_real_weather_pairs.py --split dev --per-image 1
python scripts/build_real_scene_pairs.py
python scripts/audit_pair_color.py --verify-fixed
```

上述生成命令在本次目录已经生成后会主动拒绝覆盖；复现实验需显式指定另一新输出目录。审计脚本重新基础核验会重写审计 JSON，应在其后依次执行输入复演和新版核验。

**本次没有证据证明颜色错误就是 bus 掉分的根因。** 自监督颜色分布错误与雾雨合成契约问题是需要修复的前提错误，bus 退化的定位、类别、阈值和 NMS 原因由新的逐图检测实验单独分析。
