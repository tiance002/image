> 此文件保留阶段性研究记录；当前结论、验证集口径和公开资源范围以根目录 README 与置信度公平对照报告为准。

# 2026-10-03里程碑记录

## 本次完成

1. 确认RTTS官方结构、规模和类别；
2. 记录官方短链失效及当前镜像网络超时问题；
3. 新增VOC数据质量检查工具；
4. 新增YOLO对VOC真值的独立评价工具；
5. 新增IoU和101点插值AP实现；
6. 新增对应单元测试；
7. 新增目录级批量复原及不确定性输出；
8. 使用VOC冒烟数据跑通完整评价链路。
9. 实现DCP传统去雾基线并完成三方法检测对比；
10. 修复跨目录同名输入覆盖检测结果的问题。

## 已验证命令

```powershell
python -m pytest -q
python -m ruff check .
python scripts\inspect_voc_dataset.py --images-dir <JPEGImages> --annotations-dir <Annotations>
python scripts\evaluate_voc_detection.py --images-dir <JPEGImages> --annotations-dir <Annotations>
python scripts\restore_directory.py --input-dir <JPEGImages> --output-dir <restored> --checkpoint <best.pt>
```

## 冒烟评价结果

一张公开演示图，人工建立4个VOC目标框：3个person、1个bus。

- 数据匹配：1图、1 XML、0坏图；
- person：TP=3、FP=1、FN=0、Precision=0.75、Recall=1.0；
- bus：TP=1、FP=0、FN=0、Precision=1.0、Recall=1.0；
- mAP50=1.0。

AP仍为1.0是因为额外误检的置信度低于全部正确检测；在召回达到1之前的精确率包络保持为1。这符合排序式AP定义，并不与最终Precision=0.75矛盾。

## 当前阻塞与处理

RTTS官方页面仍可验证，但Dropbox短链接被t.cn拦截；Hugging Face域名在当前机器上连续超时；一个公开Roboflow临时链接返回403。因此没有采用来源不明的替代压缩包。全部数据检查、批量复原和评价代码已经就绪，拿到合法数据后无需再改主流程。

## 下一步

1. 继续寻找可验证许可、无需账号的RTTS或DAWN下载入口；
2. 若下载仍受阻，先接入可公开获取的小规模配对去雾数据完成复原基线；
3. 增加DCP传统去雾基线，保证展示时至少有“原图—传统方法—本方法”三组对照；
4. 生成统一CSV结果汇总与展示图板。
