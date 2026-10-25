> 此文件保留阶段性研究记录；当前结论、验证集口径和公开资源范围以根目录 README 与置信度公平对照报告为准。

# 数据规范

## 1. 本地目录

数据不提交版本库。推荐结构：

```text
data/
├── raw/                     # 原始数据，只读
├── processed/               # 标准化后的图像与标签
│   ├── weather/
│   ├── clean/
│   └── labels/
├── manifests/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── reports/                 # 数据统计和泄漏检查报告
```

## 2. 清单格式

CSV必须包含：

```csv
weather_path,clean_path,scene_id,weather,severity,source
processed/weather/0001.png,processed/clean/0001.png,scene_001,rain,medium,RainCityscapes
```

字段说明：

- `weather_path`：退化图相对项目根目录的路径；
- `clean_path`：清晰真值路径，真实无配对样本可留空；
- `scene_id`：场景级唯一标识，用于防止泄漏；
- `weather`：rain、fog、rain_fog；
- `severity`：light、medium、heavy；
- `source`：数据来源名称。

## 3. 划分原则

- 按 `scene_id` 划分，不按单张图片随机划分；
- 同一视频或连续帧必须属于同一集合；
- 同一清晰图生成的不同天气版本必须属于同一集合；
- 合成参数可以跨集合重复，但基础场景不能重复；
- 真实天气测试集完全隔离。

建议比例为 70%/15%/15%，但官方划分优先于自定义比例。

## 4. 必做检查

- 路径和图像可读性；
- RGB通道和位深；
- 天气图与真值尺寸一致；
- 文件哈希重复；
- 场景ID跨集合交集；
- 类别、天气和严重度分布；
- 标注类别映射；
- 数据许可和引用方式。

## 5. 最小闭环

正式下载全部数据前，先准备 50 对样本：30 训练、10 验证、10 测试。要求能够完成：

1. 清单读取；
2. 数据增强；
3. 一轮训练；
4. 保存权重；
5. 单图推理；
6. 生成对比图。

最小闭环通过后再扩大数据规模。

