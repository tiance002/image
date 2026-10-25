# 本地源码快照范围

本轮只创建本地Git提交，没有push。保留源码、五份主要报告、官方资源核查、聚合PR/CI/画质/效率、公平性/校准/资源哈希和三张统计图。

数据、模型权重、逐图预测、原始GT坐标、完整采样与训练逐步日志、被INVALIDATED的暂存汇总不随此副本提供，仍在原始本地科研工作区。少量案例保留图像ID、类别、置信度及IoU，省略框坐标。

metrics/image_quality/benchmark聚合副本省略逐图字段；execution_manifest里的original_workspace_artifacts_sha256对应原始科研产物，PUBLICATION_MANIFEST对应实际提交副本。完整复算须准备本地数据/标注/权重及原始预测。

历史训练源码与后续NAF非整除尺寸裁剪修复已区分；三种实际尺寸的新旧前向逐位一致。Git统一LF可能改变raw字节SHA，应以normalized_lf_sha256比对源码内容，为新试验建立独立预注册清单。

本轮27项直接相关测试在具备本地YOLO权重的科研环境通过；旧105核心测试属于历史发布记录，不声称本轮重新跑过全部旧测试或UI测试。
