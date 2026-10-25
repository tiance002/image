"""Local upload interface for SafeRestore-RF."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from saferestore.artifacts import (
    create_batch_result_bundle,
    create_image_result_bundle,
    read_image_result_summary,
    summarize_image_runs,
)
from saferestore.baselines import dark_channel_dehaze
from saferestore.config import load_config
from saferestore.diagnostics import compute_image_change_diagnostics
from saferestore.models import calibrated_conservative_fusion
from saferestore.perception import (
    ADAPTIVE_GUARD_MAX_INTERVAL,
    DEFAULT_PERCEPTION_THRESHOLD,
    DetectionRecord,
    decide_perception_safe,
    schedule_perception_guard,
)
from saferestore.train import build_model
from saferestore.video import (
    create_video_result_bundle,
    render_safety_timeline,
    resize_to_max_side,
    safe_video_fps,
    simulate_realtime_schedule,
    summarize_video_traces,
    temporal_stabilize,
    write_video_trace,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v3_factorial_seed42.yaml"
CHECKPOINT_PATH = ROOT / "runs" / "v3_factorial_seed42" / "best.pt"
FUSION_CALIBRATION_PATH = ROOT / "runs" / "v3_factorial_seed42" / "fusion_calibration.json"
DETECTOR_PATH = ROOT / "yolov8n.pt"
VIDEO_OUTPUT_DIR = ROOT / "runs" / "video_demo"
IMAGE_OUTPUT_DIR = ROOT / "runs" / "image_demo"
BATCH_OUTPUT_DIR = ROOT / "runs" / "batch_demo"
DEMO_VIDEO_PATH = ROOT / "runs" / "assets" / "acdc_fog_sequence.mp4"
GUARD_EVALUATION_PATH = (
    ROOT / "runs" / "acdc_benchmark" / "perception_guard" / "summary.json"
)
PERFORMANCE_SUMMARY_PATH = ROOT / "runs" / "performance" / "video_modes" / "summary.json"
PRECISION_SUMMARY_PATH = ROOT / "runs" / "performance" / "precision" / "summary.json"
MODEL_LABEL = "V3全因子均衡版（seed 42，按验证损失选择）"
ROAD_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IMAGE_HISTORY_HEADERS = [
    "实验",
    "雨概率",
    "雾概率",
    "强度",
    "安全分数",
    "最终决策",
    "目标数",
    "总耗时ms",
]
BATCH_HEADERS = [
    "文件",
    "雨概率",
    "雾概率",
    "强度",
    "安全分数",
    "最终决策",
    "目标数",
    "总耗时ms",
]

CUSTOM_CSS = """
.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at 8% 0%, rgba(14, 165, 233, 0.12), transparent 30%),
        radial-gradient(circle at 92% 8%, rgba(37, 99, 235, 0.10), transparent 28%);
}
.sr-hero {
    position: relative;
    overflow: hidden;
    padding: 28px 32px;
    margin: 4px 0 22px;
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 22px;
    color: #f8fafc;
    background: linear-gradient(125deg, #071528 0%, #0f2a48 55%, #075985 100%);
    box-shadow: 0 18px 45px rgba(2, 8, 23, 0.18);
}
.sr-hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -70px;
    top: -120px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.16);
}
.sr-hero h1 { margin: 0 0 8px; font-size: 2.15rem; letter-spacing: -0.03em; }
.sr-hero p { margin: 0; max-width: 880px; color: #cbd5e1; line-height: 1.7; }
.sr-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.sr-badge {
    padding: 6px 11px;
    border: 1px solid rgba(186, 230, 253, 0.26);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    color: #e0f2fe;
    font-size: 0.82rem;
}
.sr-card {
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, 0.72) !important;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
}
.dark .sr-card { background: rgba(15, 23, 42, 0.74) !important; }
.sr-main-output { border: 2px solid rgba(14, 165, 233, 0.42) !important; border-radius: 16px !important; }
.sr-action button {
    min-height: 48px !important;
    border-radius: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 9px 22px rgba(2, 132, 199, 0.22);
}
.sr-stats {
    min-height: 330px;
    padding: 14px 18px !important;
    border-radius: 16px;
    border: 1px solid rgba(14, 165, 233, 0.20);
}
.sr-note {
    padding: 12px 16px;
    border-left: 4px solid #0ea5e9;
    border-radius: 8px;
    background: rgba(14, 165, 233, 0.08);
}
.sr-footer { text-align: center; color: #64748b; font-size: 0.84rem; padding: 12px 0 2px; }
"""

APP_THEME = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
)
VIDEO_PROGRESS = gr.Progress()
BATCH_PROGRESS = gr.Progress()
VIDEO_HISTORY_HEADERS = [
    "运行编号",
    "帧数",
    "最大边",
    "精度",
    "实测FPS",
    "延迟ms/帧",
    "P95延迟ms",
    "仿真丢帧",
    "闪烁下降",
    "采用候选",
    "回退输入",
]


def load_video_history() -> list[list[object]]:
    """Load recent trace summaries for the demo history table."""
    return summarize_video_traces(VIDEO_OUTPUT_DIR, limit=10)


def load_image_history() -> list[list[object]]:
    """Load recent single-image experiment summaries for the UI."""
    return summarize_image_runs(IMAGE_OUTPUT_DIR, limit=10)


def load_guard_evidence(path: Path = GUARD_EVALUATION_PATH) -> str:
    """Render the frozen ACDC gate result without overstating significance."""
    if not path.is_file():
        return "尚未生成检测安全门控评测；运行 `scripts/evaluate_perception_guard.py`。"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(payload["runtime_threshold"])
        selected = next(
            item for item in payload["thresholds"] if float(item["threshold"]) == threshold
        )
        reference = payload["reference"]
        candidate = payload["candidate"]
        difference = selected["map50_difference_vs_input"]
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError, OSError):
        return "检测安全门控评测文件无效，请重新运行评测脚本。"
    return "\n".join(
        [
            "### ACDC 200 张 fog/rain 验证图",
            "| 输出策略 | mAP50 | Recall | Precision |",
            "|---|---:|---:|---:|",
            f"| 恶劣天气输入 | {reference['map50']:.4f} | {reference['recall']:.4f} | {reference['precision']:.4f} |",
            f"| 未门控复原候选 | {candidate['map50']:.4f} | {candidate['recall']:.4f} | {candidate['precision']:.4f} |",
            f"| 检测门控（阈值 {threshold:.2f}） | {selected['map50']:.4f} | {selected['recall']:.4f} | {selected['precision']:.4f} |",
            "",
            (
                f"门控采用候选 **{selected['accept_count']}/200** 张；相对输入 mAP50 差值 "
                f"**{difference['observed_difference']:+.4f}**，bootstrap 95% CI "
                f"[{difference['ci_lower']:+.4f}, {difference['ci_upper']:+.4f}]。"
            ),
            "",
            "> 区间跨越 0：当前结果支持“限制复原造成的检测损失”，不支持“显著提升检测性能”。",
        ]
    )


def load_performance_evidence(path: Path = PERFORMANCE_SUMMARY_PATH) -> str:
    """Render the latest reproducible video-mode benchmark."""
    if not path.is_file():
        return "尚未生成视频性能基准；运行 `scripts/benchmark_video_modes.py`。"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["modes"]
    except (KeyError, TypeError, json.JSONDecodeError, OSError):
        return "视频性能基准文件无效，请重新运行基准脚本。"
    labels = {
        "safety_every_frame": "每帧安全门控",
        "risk_adaptive": "风险自适应门控",
        "balanced_every_3": "每3帧门控",
        "speed_single_detector": "关闭门控",
    }
    repeats = int(payload.get("repeats", 1))
    lines = [
        f"### 视频性能基准（{payload['frames']} 帧，最大边 {payload['max_side']}，{repeats} 轮）",
        "| 模式 | FPS | 延迟ms | 复原ms | 时序ms | 检测/门控ms | 时序/逐帧/输入 | 风险漏判 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def format_count(value: float) -> str:
            return f"{float(value):.1f}" if repeats > 1 else str(round(float(value)))

        fps = f"{row['achieved_fps']:.2f}"
        latency = f"{row['average_latency_ms']:.1f}"
        if repeats > 1:
            fps += f" ± {row.get('achieved_fps_std', 0.0):.2f}"
            latency += f" ± {row.get('average_latency_ms_std', 0.0):.1f}"
        lines.append(
            f"| {labels.get(row['mode'], row['mode'])} | {fps} "
            f"| {latency} | {row['restoration_ms']:.1f} "
            f"| {row['temporal_ms']:.1f} | {row['detection_ms']:.1f} "
            f"| {format_count(row.get('temporal_accepts', row.get('guard_accepts', 0)))}/"
            f"{format_count(row.get('frame_candidate_accepts', 0))}/"
            f"{format_count(row.get('input_fallbacks', row.get('guard_fallbacks', 0)))} "
            f"| {format_count(row.get('candidate_on_baseline_fallback', 0))} |"
        )
    lines.extend(
        [
            "",
            "> 风险漏判表示该模式输出了逐帧安全基线会回退的候选；当前速度收益不足以抵消风险，默认采用逐帧安全模式。",
        ]
    )
    return "\n".join(lines)


def load_precision_evidence(path: Path = PRECISION_SUMMARY_PATH) -> str:
    """Render the latest FP32 versus mixed-precision validation."""
    if not path.is_file():
        return "尚未生成精度基准；运行 `scripts/benchmark_precision.py`。"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fp32 = payload["fp32"]
        fp16 = payload["fp16"]
        selection = payload["selection_comparison"]
        candidate = payload["candidate_comparison"]
    except (KeyError, TypeError, json.JSONDecodeError, OSError):
        return "精度基准文件无效，请重新运行基准脚本。"
    return "\n".join(
        [
            f"### 混合精度验证（{payload['frames']} 帧，最大边 {payload['max_side']}）",
            "| 指标 | FP32 | FP16 复原 + FP32 检测 |",
            "|---|---:|---:|",
            f"| FPS | {fp32['achieved_fps']:.2f} | {fp16['achieved_fps']:.2f} |",
            f"| 平均延迟 | {fp32['average_latency_ms']:.1f} ms | {fp16['average_latency_ms']:.1f} ms |",
            f"| 复原耗时 | {fp32['restoration_ms']:.1f} ms | {fp16['restoration_ms']:.1f} ms |",
            "",
            (
                f"候选图 PSNR **{candidate['candidate_psnr_db']:.2f} dB**，"
                f"门控选择不一致 **{selection['selection_disagreements']}/"
                f"{selection['compared_frames']}** 帧，"
                f"总体加速比 **{payload['speedup']:.3f}x**。"
            ),
            "",
            "> 当前加速收益很小，因此 FP16 保留为实验选项，默认仍采用 FP32。",
        ]
    )


def load_evidence_bundle() -> tuple[str, str, str]:
    return (
        load_guard_evidence(),
        load_performance_evidence(),
        load_precision_evidence(),
    )


def configure_local_network() -> None:
    """Keep Gradio's startup self-check away from system HTTP proxies."""
    required = {"127.0.0.1", "localhost"}
    existing = {
        item.strip()
        for variable in ("NO_PROXY", "no_proxy")
        for item in os.environ.get(variable, "").split(",")
        if item.strip()
    }
    value = ",".join(sorted(existing | required))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


class DemoModels:
    """Load restoration and detection models once for repeated uploads."""

    def __init__(self) -> None:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Restoration checkpoint is missing: {CHECKPOINT_PATH}. "
                "Run the demo training command in README first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = load_config(CONFIG_PATH)
        self.restorer = build_model(config).to(self.device).eval()
        payload = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
        self.restorer.load_state_dict(payload["model"])
        self.detector = YOLO(str(DETECTOR_PATH if DETECTOR_PATH.is_file() else "yolov8n.pt"))
        self.fusion_parameters: dict[str, float] | None = None
        if FUSION_CALIBRATION_PATH.is_file():
            calibration = json.loads(FUSION_CALIBRATION_PATH.read_text(encoding="utf-8"))
            self.fusion_parameters = {
                name: float(value) for name, value in calibration["parameters"].items()
            }

    def warm_up(self) -> None:
        """Warm CUDA and detector kernels before the user's first timed run."""
        sample_path = ROOT / "runs" / "assets" / "fog.jpg"
        if not sample_path.is_file():
            return
        with Image.open(sample_path) as image:
            sample = np.asarray(image.convert("RGB"))
        sample = resize_to_max_side(sample, 640)
        for _ in range(2):
            _, _, _, candidate, _ = self._restore_rgb(sample)
            self._predict_rgb(sample)
            self._predict_rgb(candidate)
        if self.device.type == "cuda":
            # Avoid charging first-use autocast kernel setup to the user's run.
            self._restore_rgb(sample, use_half=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _restore_rgb(self, input_image: np.ndarray, use_half: bool = False):
        """Run restoration and calibrated fusion for one RGB frame."""
        image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
        tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
        use_half = bool(use_half and self.device.type == "cuda")
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=use_half,
        ):
            outputs = self.restorer(tensor)
        restored = np.asarray(TF.to_pil_image(outputs["restored"][0].float().cpu()))
        if self.fusion_parameters is not None:
            conservative_tensor, preservation_weight = calibrated_conservative_fusion(
                outputs["restored"],
                tensor,
                outputs["uncertainty"],
                outputs["degradation"][:, 2],
                **self.fusion_parameters,
            )
        else:
            conservative_tensor = outputs["conservative"]
            preservation_weight = outputs["uncertainty"]
        conservative = np.asarray(TF.to_pil_image(conservative_tensor[0].float().cpu()))
        return image, outputs, restored, conservative, preservation_weight

    def _predict_rgb(self, image: np.ndarray, use_half: bool = False):
        """Run the frozen road detector for one RGB image."""
        return self.detector.predict(
            source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            imgsz=640,
            conf=0.25,
            iou=0.7,
            device=0 if self.device.type == "cuda" else "cpu",
            classes=list(ROAD_CLASSES),
            verbose=False,
        )[0]

    @staticmethod
    def _detection_records(result) -> list[DetectionRecord]:
        records: list[DetectionRecord] = []
        if result.boxes is not None:
            for box in result.boxes:
                records.append(
                    DetectionRecord(
                        class_id=int(box.cls.item()),
                        confidence=float(box.conf.item()),
                        box=box.xyxy[0].detach().cpu().numpy(),
                    )
                )
        return records

    @staticmethod
    def _render_detection(image: np.ndarray, result) -> np.ndarray:
        annotated_bgr = result.plot(img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _summarize_detection(result):
        counts = {name: 0 for name in ROAD_CLASSES.values()}
        confidences: list[float] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id in ROAD_CLASSES:
                    counts[ROAD_CLASSES[class_id]] += 1
                    confidences.append(float(box.conf.item()))
        return counts, confidences

    def _detect_rgb(self, image: np.ndarray, use_half: bool = False):
        """Detect and render road users for one RGB image."""
        result = self._predict_rgb(image, use_half=use_half)
        counts, confidences = self._summarize_detection(result)
        return self._render_detection(image, result), counts, confidences, result

    def _apply_perception_guard(
        self,
        input_rgb: np.ndarray,
        candidate_rgb: np.ndarray,
        use_half: bool = False,
    ):
        """Select candidate or input using frozen-detector consistency."""
        input_result = self._predict_rgb(input_rgb, use_half=use_half)
        candidate_result = self._predict_rgb(candidate_rgb, use_half=use_half)
        decision = decide_perception_safe(
            self._detection_records(input_result),
            self._detection_records(candidate_result),
            minimum_score=DEFAULT_PERCEPTION_THRESHOLD,
        )
        if decision.accept_candidate:
            selected_image, selected_result = candidate_rgb, candidate_result
        else:
            selected_image, selected_result = input_rgb, input_result
        counts, confidences = self._summarize_detection(selected_result)
        detected = self._render_detection(selected_image, selected_result)
        return selected_image, detected, counts, confidences, decision

    def _apply_video_perception_guard(
        self,
        input_rgb: np.ndarray,
        frame_candidate_rgb: np.ndarray,
        temporal_candidate_rgb: np.ndarray,
        use_half: bool = False,
    ):
        """Select temporal candidate, frame candidate, or input in that order."""
        input_result = self._predict_rgb(input_rgb, use_half=use_half)
        input_records = self._detection_records(input_result)
        temporal_result = self._predict_rgb(temporal_candidate_rgb, use_half=use_half)
        temporal_decision = decide_perception_safe(
            input_records,
            self._detection_records(temporal_result),
            minimum_score=DEFAULT_PERCEPTION_THRESHOLD,
        )
        if temporal_decision.accept_candidate:
            selection = "temporal_candidate"
            selected_image, selected_result = temporal_candidate_rgb, temporal_result
            selected_decision = temporal_decision
        elif np.array_equal(frame_candidate_rgb, temporal_candidate_rgb):
            selection = "input_fallback"
            selected_image, selected_result = input_rgb, input_result
            selected_decision = temporal_decision
        else:
            frame_result = self._predict_rgb(frame_candidate_rgb, use_half=use_half)
            frame_decision = decide_perception_safe(
                input_records,
                self._detection_records(frame_result),
                minimum_score=DEFAULT_PERCEPTION_THRESHOLD,
            )
            if frame_decision.accept_candidate:
                selection = "frame_candidate"
                selected_image, selected_result = frame_candidate_rgb, frame_result
            else:
                selection = "input_fallback"
                selected_image, selected_result = input_rgb, input_result
            selected_decision = frame_decision
        counts, confidences = self._summarize_detection(selected_result)
        detected = self._render_detection(selected_image, selected_result)
        return (
            selected_image,
            detected,
            counts,
            confidences,
            selected_decision,
            selection,
            temporal_decision.safety_score,
        )

    def process(self, input_image: np.ndarray | None):
        if input_image is None:
            raise gr.Error("请先上传一张 JPG、JPEG 或 PNG 图片。")
        pipeline_started = time.perf_counter()
        restoration_started = time.perf_counter()
        image, outputs, restored, conservative, preservation_weight = self._restore_rgb(
            input_image
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        restoration_ms = (time.perf_counter() - restoration_started) * 1000.0
        uncertainty_gray = np.asarray(TF.to_pil_image(outputs["uncertainty"][0].cpu()))
        uncertainty_values = outputs["uncertainty"][0, 0].detach().cpu().numpy()
        uncertainty_heatmap = cv2.applyColorMap(uncertainty_gray, cv2.COLORMAP_TURBO)
        uncertainty_heatmap = cv2.cvtColor(uncertainty_heatmap, cv2.COLOR_BGR2RGB)
        baseline_started = time.perf_counter()
        dcp, _ = dark_channel_dehaze(np.asarray(image))
        baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
        guard_started = time.perf_counter()
        final_image, detected, counts, confidences, decision = self._apply_perception_guard(
            np.asarray(image), conservative
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        guard_ms = (time.perf_counter() - guard_started) * 1000.0
        pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
        candidate_change = compute_image_change_diagnostics(np.asarray(image), conservative)
        final_change = compute_image_change_diagnostics(np.asarray(image), final_image)
        count_text = ", ".join(f"{name}: {count}" for name, count in counts.items() if count)
        if not count_text:
            count_text = "未检测到设定的道路目标"
        degradation = outputs["degradation"][0].detach().cpu()
        stat_lines = [
            "### 本次处理结果",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 运行设备 | `{self.device}` |",
            f"| 模型复原耗时 | {restoration_ms:.1f} ms |",
            f"| DCP 基线耗时 | {baseline_ms:.1f} ms |",
            f"| 检测与安全门控耗时 | {guard_ms:.1f} ms |",
            f"| 总处理链路耗时 | {pipeline_ms:.1f} ms |",
            f"| 雨概率 | {float(degradation[0]):.3f} |",
            f"| 雾概率 | {float(degradation[1]):.3f} |",
            f"| 预测退化强度 | {float(degradation[2]):.3f} |",
            f"| 雨雾混合概率 | {float(degradation[3]):.3f} |",
            f"| 平均不确定性 | {uncertainty_values.mean():.3f} |",
            f"| 90%分位不确定性 | {np.quantile(uncertainty_values, 0.9):.3f} |",
            f"| 高风险像素比例（≥0.7） | {np.mean(uncertainty_values >= 0.7):.1%} |",
            f"| 平均输入保留权重 | {float(preservation_weight.mean()):.3f} |",
            f"| 候选平均像素改动 | {candidate_change.mean_absolute_change:.3f} |",
            f"| 候选显著改动像素（≥20/255） | {candidate_change.significant_change_ratio:.1%} |",
            f"| 候选梯度结构一致性 | {candidate_change.gradient_cosine_similarity:.3f} |",
            f"| 最终输出平均像素改动 | {final_change.mean_absolute_change:.3f} |",
            f"| 检测安全分数 | {decision.safety_score:.3f} |",
            f"| 检测安全阈值 | {DEFAULT_PERCEPTION_THRESHOLD:.2f} |",
            f"| 输入参考目标保持率 | {decision.reference_retention:.1%} |",
            f"| 参考目标匹配 | {decision.matched_reference}/{decision.reference_count} |",
            f"| 感知门控决策 | **{'采用复原候选' if decision.accept_candidate else '回退输入图'}** |",
            "",
            f"**决策说明：** {decision.reason}",
            "",
            f"**道路目标：** {count_text}",
        ]
        if confidences:
            stat_lines.append(f"\n**平均检测置信度：** {np.mean(confidences):.3f}")
        stat_lines.extend(
            [
                "",
                f"<small>模型：{MODEL_LABEL}</small>",
            ]
        )
        stats = "\n\n".join(stat_lines)
        degradation_values = [float(value) for value in degradation]
        metrics = {
            "schema_version": 1,
            "model": MODEL_LABEL,
            "device": str(self.device),
            "degradation": {
                "rain_probability": degradation_values[0],
                "fog_probability": degradation_values[1],
                "severity": degradation_values[2],
                "mixed_probability": degradation_values[3],
            },
            "uncertainty": {
                "mean": float(uncertainty_values.mean()),
                "p90": float(np.quantile(uncertainty_values, 0.9)),
                "high_risk_ratio": float(np.mean(uncertainty_values >= 0.7)),
            },
            "input_preservation_mean": float(preservation_weight.mean()),
            "safety": {
                "threshold": DEFAULT_PERCEPTION_THRESHOLD,
                "score": decision.safety_score,
                "accepted_restoration": decision.accept_candidate,
                "reference_retention": decision.reference_retention,
                "matched_reference": decision.matched_reference,
                "reference_count": decision.reference_count,
                "reason": decision.reason,
            },
            "detections": {
                "counts": counts,
                "mean_confidence": float(np.mean(confidences)) if confidences else None,
            },
            "timing_ms": {
                "restoration": restoration_ms,
                "dcp_baseline": baseline_ms,
                "detection_and_guard": guard_ms,
                "pipeline_total": pipeline_ms,
            },
            "change_diagnostics": {
                "interpretation": "Modification relative to input; not a ground-truth quality score.",
                "candidate": {
                    "mean_absolute_change": candidate_change.mean_absolute_change,
                    "p95_absolute_change": candidate_change.p95_absolute_change,
                    "significant_change_ratio": candidate_change.significant_change_ratio,
                    "gradient_cosine_similarity": candidate_change.gradient_cosine_similarity,
                },
                "final_output": {
                    "mean_absolute_change": final_change.mean_absolute_change,
                    "p95_absolute_change": final_change.p95_absolute_change,
                    "significant_change_ratio": final_change.significant_change_ratio,
                    "gradient_cosine_similarity": final_change.gradient_cosine_similarity,
                },
            },
        }
        experiment_id = f"saferestore_image_{uuid.uuid4().hex[:10]}"
        image_bundle = create_image_result_bundle(
            IMAGE_OUTPUT_DIR,
            experiment_id,
            {
                "input": np.asarray(image),
                "direct_restoration": restored,
                "conservative_output": final_image,
                "uncertainty_heatmap": uncertainty_heatmap,
                "dcp_baseline": dcp,
                "detection_result": detected,
                "candidate_change_heatmap": candidate_change.heatmap,
            },
            metrics,
        )
        return (
            restored,
            final_image,
            uncertainty_heatmap,
            candidate_change.heatmap,
            dcp,
            detected,
            stats,
            str(image_bundle),
        )

    def process_video(
        self,
        video_path: str | None,
        max_frames: int,
        output_fps: float,
        max_side: int,
        temporal_strength: float,
        perception_guard: bool,
        guard_interval: int | str,
        inference_precision: str = "fp32",
        progress=VIDEO_PROGRESS,
    ):
        """Run the image pipeline frame-by-frame for an offline driving simulation."""
        if not video_path:
            raise gr.Error("请先上传 MP4 视频或使用预载的 ACDC 序列。")
        input_fd, input_name = tempfile.mkstemp(suffix=Path(video_path).suffix or ".mp4")
        os.close(input_fd)
        temporary_input = Path(input_name)
        shutil.copy2(video_path, temporary_input)
        capture = cv2.VideoCapture(str(temporary_input))
        if not capture.isOpened():
            temporary_input.unlink(missing_ok=True)
            raise gr.Error("视频无法读取，请上传常见编码的 MP4 文件。")

        max_frames = max(1, int(max_frames))
        max_side = max(320, int(max_side))
        source_fps = safe_video_fps(capture.get(cv2.CAP_PROP_FPS))
        requested_fps = safe_video_fps(float(output_fps), source_fps)
        requested_precision = str(inference_precision).lower()
        use_half = requested_precision == "fp16" and self.device.type == "cuda"
        effective_precision = "fp16" if use_half else "fp32"
        source_total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        planned_frames = min(max_frames, source_total) if source_total else max_frames
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = VIDEO_OUTPUT_DIR / f"saferestore_{uuid.uuid4().hex[:10]}.mp4"
        output_fd, output_name = tempfile.mkstemp(suffix=".mp4")
        os.close(output_fd)
        temporary_output = Path(output_name)
        writer: cv2.VideoWriter | None = None
        processed = 0
        total_seconds = 0.0
        uncertainty_means: list[float] = []
        preservation_means: list[float] = []
        object_counts = {name: 0 for name in ROAD_CLASSES.values()}
        temporal_weights: list[float] = []
        raw_temporal_errors: list[float] = []
        stabilized_temporal_errors: list[float] = []
        scene_cuts = 0
        previous_input: np.ndarray | None = None
        previous_output: np.ndarray | None = None
        last_guard_selection = "temporal_candidate"
        last_safety_score = 1.0
        last_temporal_safety_score = 1.0
        guard_accepts = 0
        guard_fallbacks = 0
        temporal_accepts = 0
        frame_candidate_accepts = 0
        input_fallbacks = 0
        guard_evaluations = 0
        perception_scores: list[float] = []
        frame_records: list[dict[str, object]] = []
        restoration_times_ms: list[float] = []
        temporal_times_ms: list[float] = []
        detection_times_ms: list[float] = []
        frame_latencies_ms: list[float] = []
        adaptive_guard = str(guard_interval).lower() == "adaptive"
        fixed_guard_interval = (
            ADAPTIVE_GUARD_MAX_INTERVAL
            if adaptive_guard
            else max(1, int(guard_interval))
        )
        frames_since_guard = fixed_guard_interval
        guard_trigger_counts: dict[str, int] = {}

        try:
            while processed < max_frames:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_bgr = resize_to_max_side(frame_bgr, max_side)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                started = time.perf_counter()
                stage_started = time.perf_counter()
                _, outputs, _, conservative, preservation_weight = self._restore_rgb(
                    frame_rgb,
                    use_half=use_half,
                )
                restoration_times_ms.append((time.perf_counter() - stage_started) * 1000.0)
                stabilized = conservative
                temporal = None
                stage_started = time.perf_counter()
                if previous_input is not None and previous_output is not None:
                    temporal = temporal_stabilize(
                        frame_rgb,
                        conservative,
                        previous_input,
                        previous_output,
                        strength=float(temporal_strength),
                    )
                    stabilized = temporal.image
                    if temporal.scene_cut:
                        scene_cuts += 1
                    else:
                        temporal_weights.append(temporal.mean_blend_weight)
                        raw_temporal_errors.append(temporal.raw_temporal_error)
                        stabilized_temporal_errors.append(temporal.stabilized_temporal_error)
                temporal_times_ms.append((time.perf_counter() - stage_started) * 1000.0)
                decision = None
                scene_cut = bool(temporal.scene_cut) if temporal is not None else False
                if adaptive_guard:
                    guard_evaluated, guard_trigger = schedule_perception_guard(
                        processed,
                        frames_since_guard,
                        last_safety_score,
                        last_guard_selection,
                        scene_cut,
                    )
                    guard_evaluated = perception_guard and guard_evaluated
                else:
                    guard_evaluated = (
                        perception_guard and processed % fixed_guard_interval == 0
                    )
                    guard_trigger = "fixed_interval" if guard_evaluated else "fixed_reuse"
                stage_started = time.perf_counter()
                if guard_evaluated:
                    (
                        final_image,
                        detected,
                        counts,
                        _,
                        decision,
                        selection,
                        temporal_safety_score,
                    ) = self._apply_video_perception_guard(
                        frame_rgb,
                        conservative,
                        stabilized,
                        use_half=use_half,
                    )
                    guard_evaluations += 1
                    perception_scores.append(decision.safety_score)
                    last_guard_selection = selection
                    last_safety_score = decision.safety_score
                    last_temporal_safety_score = temporal_safety_score
                    frames_since_guard = 0
                    guard_trigger_counts[guard_trigger] = (
                        guard_trigger_counts.get(guard_trigger, 0) + 1
                    )
                elif perception_guard:
                    selection = last_guard_selection
                    if selection == "temporal_candidate":
                        final_image = stabilized
                    elif selection == "frame_candidate":
                        final_image = conservative
                    else:
                        final_image = frame_rgb
                    detected, counts, _, _ = self._detect_rgb(
                        final_image,
                        use_half=use_half,
                    )
                else:
                    selection = "temporal_candidate"
                    final_image = stabilized
                    detected, counts, _, _ = self._detect_rgb(
                        final_image,
                        use_half=use_half,
                    )
                frames_since_guard += 1
                detection_times_ms.append((time.perf_counter() - stage_started) * 1000.0)
                if perception_guard:
                    if selection == "temporal_candidate":
                        temporal_accepts += 1
                        guard_accepts += 1
                    elif selection == "frame_candidate":
                        frame_candidate_accepts += 1
                        guard_accepts += 1
                    else:
                        input_fallbacks += 1
                        guard_fallbacks += 1
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                frame_latency_ms = (time.perf_counter() - started) * 1000.0
                total_seconds += frame_latency_ms / 1000.0
                frame_latencies_ms.append(frame_latency_ms)
                uncertainty_means.append(float(outputs["uncertainty"].mean().item()))
                preservation_means.append(float(preservation_weight.mean().item()))
                for name, count in counts.items():
                    object_counts[name] += count
                previous_input = frame_rgb
                # Keep the restoration stream independent from the final safety fallback.
                # Otherwise one fallback would inject a raw frame into the next candidate.
                previous_output = stabilized

                frame_records.append(
                    {
                        "frame": processed + 1,
                        "uncertainty_mean": round(float(outputs["uncertainty"].mean().item()), 6),
                        "input_preservation_mean": round(float(preservation_weight.mean().item()), 6),
                        "temporal_blend_weight": round(
                            temporal.mean_blend_weight if temporal is not None else 0.0,
                            6,
                        ),
                        "scene_cut": bool(temporal.scene_cut) if temporal is not None else False,
                        "perception_safety_score": round(last_safety_score, 6)
                        if perception_guard
                        else None,
                        "temporal_safety_score": round(last_temporal_safety_score, 6)
                        if perception_guard
                        else None,
                        "reference_retention": round(decision.reference_retention, 6)
                        if decision is not None
                        else None,
                        "guard_evaluated": guard_evaluated,
                        "guard_trigger": guard_trigger if perception_guard else "disabled",
                        "guard_selection": selection,
                        "candidate_accepted": selection != "input_fallback",
                        "detected_objects": int(sum(counts.values())),
                        "processing_latency_ms": round(frame_latency_ms, 3),
                    }
                )

                output_bgr = cv2.cvtColor(detected, cv2.COLOR_RGB2BGR)
                if writer is None:
                    height, width = output_bgr.shape[:2]
                    writer = cv2.VideoWriter(
                        str(temporary_output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        requested_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise gr.Error("无法创建结果视频，请检查 runs/video_demo 目录权限。")
                writer.write(output_bgr)
                processed += 1
                progress(processed / max(planned_frames, 1), desc=f"正在处理第 {processed} 帧")
        except Exception:
            if writer is not None:
                writer.release()
                writer = None
            temporary_output.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            temporary_input.unlink(missing_ok=True)
            if writer is not None:
                writer.release()

        if processed == 0:
            temporary_output.unlink(missing_ok=True)
            raise gr.Error("视频中没有可处理的画面。")
        shutil.copy2(temporary_output, output_path)
        temporary_output.unlink(missing_ok=True)

        average_ms = 1000.0 * total_seconds / processed
        achieved_fps = processed / max(total_seconds, 1e-9)
        counts_text = "、".join(
            f"{name} {count}" for name, count in object_counts.items() if count
        ) or "未检测到设定道路目标"
        raw_flicker = float(np.mean(raw_temporal_errors)) if raw_temporal_errors else 0.0
        stable_flicker = (
            float(np.mean(stabilized_temporal_errors)) if stabilized_temporal_errors else 0.0
        )
        flicker_reduction = (
            1.0 - stable_flicker / raw_flicker if raw_flicker > 1e-9 else 0.0
        )
        realtime = simulate_realtime_schedule(frame_latencies_ms, requested_fps)
        if realtime.simulated_drop_count > 0:
            realtime_label = "未达到（仿真会丢帧）"
        elif realtime.deadline_miss_count > 0:
            realtime_label = "边缘可达（存在超时帧）"
        else:
            realtime_label = "达到"
        timeline_path = output_path.with_name(f"{output_path.stem}_safety_timeline.png")
        render_safety_timeline(
            frame_records,
            timeline_path,
            threshold=DEFAULT_PERCEPTION_THRESHOLD,
        )
        trace_payload = {
            "schema_version": 1,
            "input_video": str(video_path),
            "output_video": str(output_path),
            "artifacts": {"safety_timeline": str(timeline_path)},
            "settings": {
                "max_frames": max_frames,
                "output_fps": requested_fps,
                "max_side": max_side,
                "temporal_strength": float(temporal_strength),
                "perception_guard": bool(perception_guard),
                "perception_threshold": DEFAULT_PERCEPTION_THRESHOLD,
                "guard_schedule": "adaptive" if adaptive_guard else "fixed",
                "guard_interval": fixed_guard_interval,
                "requested_precision": requested_precision,
                "effective_precision": effective_precision,
            },
            "summary": {
                "processed_frames": processed,
                "average_latency_ms": average_ms,
                "achieved_fps": achieved_fps,
                "raw_temporal_error": raw_flicker,
                "stabilized_temporal_error": stable_flicker,
                "guard_accepts": guard_accepts,
                "guard_fallbacks": guard_fallbacks,
                "temporal_accepts": temporal_accepts,
                "frame_candidate_accepts": frame_candidate_accepts,
                "input_fallbacks": input_fallbacks,
                "guard_evaluations": guard_evaluations,
                "guard_trigger_counts": guard_trigger_counts,
                "restoration_ms": float(np.mean(restoration_times_ms)),
                "temporal_ms": float(np.mean(temporal_times_ms)),
                "detection_ms": float(np.mean(detection_times_ms)),
                "latency_p50_ms": realtime.latency_p50_ms,
                "latency_p95_ms": realtime.latency_p95_ms,
                "frame_budget_ms": realtime.frame_budget_ms,
                "deadline_miss_count": realtime.deadline_miss_count,
                "deadline_miss_ratio": realtime.deadline_miss_ratio,
                "simulated_keep_count": realtime.simulated_keep_count,
                "simulated_drop_count": realtime.simulated_drop_count,
                "simulated_effective_fps": realtime.simulated_effective_fps,
            },
        }
        trace_json, trace_csv = write_video_trace(output_path, trace_payload, frame_records)
        result_bundle = create_video_result_bundle(
            output_path,
            trace_json,
            trace_csv,
            timeline_path,
        )
        stats = "\n\n".join(
            [
                "### 视频仿真结果",
                "| 指标 | 结果 |",
                "|---|---:|",
                f"| 已处理帧数 | {processed} |",
                f"| 输入视频帧率 | {source_fps:.1f} FPS |",
                f"| 结果视频帧率 | {requested_fps:.1f} FPS |",
                f"| 平均处理延迟 | {average_ms:.1f} ms/帧 |",
                f"| 实测处理速度 | {achieved_fps:.2f} FPS |",
                f"| 单帧延迟 P50 / P95 | {realtime.latency_p50_ms:.1f} / {realtime.latency_p95_ms:.1f} ms |",
                f"| 仿真实时帧预算 | {realtime.frame_budget_ms:.1f} ms（{requested_fps:.1f} FPS） |",
                f"| 超过帧预算 | {realtime.deadline_miss_count}/{processed} 帧（{realtime.deadline_miss_ratio:.1%}） |",
                f"| 忙碌丢帧仿真 | 保留 {realtime.simulated_keep_count} / 丢弃 {realtime.simulated_drop_count} 帧 |",
                f"| 仿真有效输出速度 | {realtime.simulated_effective_fps:.2f} FPS |",
                f"| 推理精度 | 复原 {effective_precision.upper()} / 检测 FP32 |",
                f"| 复原平均耗时 | {np.mean(restoration_times_ms):.1f} ms/帧 |",
                f"| 时序稳定平均耗时 | {np.mean(temporal_times_ms):.1f} ms/帧 |",
                f"| 检测与门控平均耗时 | {np.mean(detection_times_ms):.1f} ms/帧 |",
                f"| 实时可行性判定 | **{realtime_label}** |",
                f"| 平均不确定性 | {np.mean(uncertainty_means):.3f} |",
                f"| 平均输入保留权重 | {np.mean(preservation_means):.3f} |",
                f"| 时序融合实际平均权重 | {np.mean(temporal_weights) if temporal_weights else 0.0:.3f} |",
                f"| 稳定前运动补偿闪烁 | {raw_flicker:.4f} |",
                f"| 稳定后运动补偿闪烁 | {stable_flicker:.4f} |",
                f"| 闪烁指标下降 | {flicker_reduction:.1%} |",
                f"| 检测到的场景切换 | {scene_cuts} |",
                f"| 采用时序稳定候选 | {temporal_accepts} 帧 |",
                f"| 绕过时序、采用逐帧候选 | {frame_candidate_accepts} 帧 |",
                f"| 回退输入 | {input_fallbacks} 帧 |",
                f"| 门控调度 | {'风险自适应（最长每 3 帧）' if adaptive_guard else f'固定每 {fixed_guard_interval} 帧'} |",
                f"| 实际分层门控评估 | {guard_evaluations} 次 |",
                f"| 平均检测安全分数 | {np.mean(perception_scores) if perception_scores else 1.0:.3f} |",
                f"| 检测安全阈值 | {DEFAULT_PERCEPTION_THRESHOLD:.2f} |",
                "",
                f"**累计检测：** {counts_text}",
                "",
                "<small>忙碌丢帧是根据实测逐帧耗时进行的单工作线程离线调度仿真；结果 MP4 仍保留全部处理帧，便于画质对比。安全阈值由 ACDC 验证集探索确定，不支持宣称显著提升检测性能。</small>",
            ]
        )
        return (
            str(output_path),
            stats,
            str(timeline_path),
            [str(result_bundle), str(trace_json), str(trace_csv), str(timeline_path)],
        )

    def process_batch(self, uploaded_files: list[str] | None, progress=BATCH_PROGRESS):
        """Process a small image collection with the frozen interactive protocol."""
        if not uploaded_files:
            raise gr.Error("请先上传至少一张 JPG 或 PNG 图片。")
        if len(uploaded_files) > 12:
            raise gr.Error("为避免显存与等待时间失控，单次最多处理 12 张图片。")
        item_bundles: list[Path] = []
        summary_rows: list[dict[str, object]] = []
        for index, uploaded in enumerate(uploaded_files):
            source_path = Path(str(uploaded))
            try:
                with Image.open(source_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                results = self.process(rgb)
                item_bundle = Path(results[-1])
                summary = read_image_result_summary(item_bundle, source_path.name)
            except Exception as error:
                raise gr.Error(f"批量处理失败：{source_path.name}：{error}") from error
            item_bundles.append(item_bundle)
            summary_rows.append(summary)
            progress((index + 1) / len(uploaded_files), desc=f"正在处理 {source_path.name}")

        batch_id = f"saferestore_batch_{uuid.uuid4().hex[:10]}"
        batch_bundle, summary_csv = create_batch_result_bundle(
            BATCH_OUTPUT_DIR,
            batch_id,
            item_bundles,
            summary_rows,
        )
        accepted = sum(row["decision"] == "accepted" for row in summary_rows)
        average_ms = float(
            np.mean([float(row["pipeline_total_ms"]) for row in summary_rows])
        )
        detected_objects = sum(int(row["detected_objects"]) for row in summary_rows)
        table_rows = [
            [
                row["source_file"],
                row["rain_probability"],
                row["fog_probability"],
                row["severity"],
                row["safety_score"],
                "采用复原" if row["decision"] == "accepted" else "回退输入",
                row["detected_objects"],
                row["pipeline_total_ms"],
            ]
            for row in summary_rows
        ]
        stats = "\n\n".join(
            [
                "### 批量实验结果",
                f"- 成功处理：**{len(summary_rows)} 张**",
                f"- 采用复原 / 回退输入：**{accepted} / {len(summary_rows) - accepted}**",
                f"- 平均单图链路耗时：**{average_ms:.1f} ms**",
                f"- 累计道路目标：**{detected_objects} 个**",
            ]
        )
        return table_rows, stats, [str(batch_bundle), str(summary_csv)]


def build_interface(models: DemoModels) -> gr.Blocks:
    with gr.Blocks(title="SafeRestore-RF｜恶劣天气安全感知复原") as interface:
        gr.HTML(
            """
            <section class="sr-hero">
              <h1>SafeRestore-RF</h1>
              <p>面向道路安全感知的保真型恶劣天气图像复原。一次查看直接复原、风险约束融合、传统基线与道路目标检测结果。</p>
              <div class="sr-badges">
                <span class="sr-badge">雨 / 雾 / 混合退化</span>
                <span class="sr-badge">像素级不确定性</span>
                <span class="sr-badge">保真融合</span>
                <span class="sr-badge">道路目标检测</span>
              </div>
            </section>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=7, elem_classes=["sr-card"]):
                gr.Markdown("## ① 选择测试图片")
                gr.Markdown("上传一张 JPG、JPEG 或 PNG；也可以直接使用预载的雾天样例。")
                input_image = gr.Image(
                    value=str(ROOT / "runs" / "assets" / "fog.jpg"),
                    label="恶劣天气道路图像",
                    type="numpy",
                    image_mode="RGB",
                    height=390,
                )
                run_button = gr.Button(
                    "开始复原与安全检测",
                    variant="primary",
                    elem_classes=["sr-action"],
                )
            with gr.Column(scale=5, elem_classes=["sr-card"]):
                gr.Markdown("## ② 查看运行指标")
                stats = gr.Markdown(
                    """
                    <div class="sr-note">
                    <b>等待处理</b><br>
                    点击“开始复原与安全检测”后，这里会显示天气判断、不确定性、原图保留比例和目标检测统计。
                    </div>
                    """,
                    elem_classes=["sr-stats"],
                )
                with gr.Accordion("如何理解输出？", open=False):
                    gr.Markdown(
                        "- **安全融合图**：根据退化强度和不确定性，在原图与复原图之间进行有界融合。\n"
                        "- **风险热力图**：颜色越热表示模型对该区域的复原越不确定。\n"
                        "- **检测结果**：当前使用冻结的 YOLOv8n 检测道路目标。"
                    )
                image_bundle = gr.File(
                    label="单图完整实验包 ZIP",
                    interactive=False,
                )

        gr.Markdown("## ③ 核心结果")
        with gr.Tabs():
            with gr.Tab("安全感知结果"):
                with gr.Row(equal_height=True):
                    conservative = gr.Image(
                        label="验证集校准保真融合",
                        height=420,
                        elem_classes=["sr-main-output"],
                    )
                    detected = gr.Image(
                        label="融合后的道路目标检测",
                        height=420,
                        elem_classes=["sr-main-output"],
                    )
                gr.Markdown(
                    "优先观察这一页：左侧是系统最终复原输出，右侧是同一图像上的行人、车辆等检测结果。"
                )
            with gr.Tab("复原方法对比"):
                with gr.Row(equal_height=True):
                    restored = gr.Image(label="学习式直接复原", height=380)
                    dcp = gr.Image(label="DCP传统去雾基线", height=380)
                gr.Markdown("用于比较学习式复原与传统暗通道先验方法，不代表任一方法必然提升检测性能。")
            with gr.Tab("不确定性分析"):
                with gr.Row(equal_height=True):
                    uncertainty = gr.Image(label="像素级不确定性热力图", height=430)
                    change_heatmap = gr.Image(label="候选复原改动热力图", height=430)
                gr.Markdown(
                    "左图表示模型自身的不确定性；右图表示候选复原相对输入改动的位置和幅度。"
                    "改动幅度不是无参考画质分数，需要结合检测安全决策理解。"
                )

        with gr.Accordion("最近单图实验对比", open=False):
            image_history_table = gr.Dataframe(
                headers=IMAGE_HISTORY_HEADERS,
                value=load_image_history(),
                interactive=False,
                wrap=True,
                label="最近 10 次有效单图实验",
            )
            refresh_image_history = gr.Button("刷新单图实验历史", size="sm")

        gr.Markdown("## ④ 多图片批量实验")
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["sr-card"]):
                batch_input = gr.File(
                    label="批量上传道路图片（最多 12 张）",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                batch_button = gr.Button(
                    "开始批量复原与安全检测",
                    variant="primary",
                    elem_classes=["sr-action"],
                )
                batch_stats = gr.Markdown("等待上传多张图片。")
                batch_files = gr.File(
                    label="批量总 ZIP 与汇总 CSV",
                    file_count="multiple",
                    interactive=False,
                )
            with gr.Column(scale=9, elem_classes=["sr-card"]):
                batch_table = gr.Dataframe(
                    headers=BATCH_HEADERS,
                    interactive=False,
                    wrap=True,
                    label="逐图安全决策汇总",
                )

        gr.Markdown("## ⑤ 无摄像头视频仿真")
        gr.Markdown(
            "上传行车 MP4，或直接使用预载的 ACDC 雾天序列。系统逐帧执行复原、保真融合和道路目标检测，"
            "生成带检测框的结果视频，并测量当前设备的平均延迟与吞吐速度。"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=7, elem_classes=["sr-card"]):
                video_input = gr.Video(
                    value=str(DEMO_VIDEO_PATH) if DEMO_VIDEO_PATH.is_file() else None,
                    label="输入视频 / ACDC 序列回放",
                    sources=["upload"],
                    format="mp4",
                    height=390,
                )
                with gr.Row():
                    max_frames = gr.Slider(
                        minimum=10,
                        maximum=120,
                        value=30,
                        step=10,
                        label="最多处理帧数",
                    )
                    output_fps = gr.Slider(
                        minimum=5,
                        maximum=25,
                        value=10,
                        step=1,
                        label="仿真传感器 / 结果视频帧率",
                        info="同时作为实时帧预算；例如 25 FPS 对应每帧 40 ms。",
                    )
                    max_side = gr.Dropdown(
                        choices=[480, 640, 960],
                        value=640,
                        label="最大边分辨率",
                    )
                temporal_strength = gr.Slider(
                    minimum=0.0,
                    maximum=0.35,
                    value=0.18,
                    step=0.01,
                    label="时序稳定强度（0=关闭，建议 0.12–0.22）",
                    info="仅在运动补偿后仍可信的静止区域复用上一帧，动态目标区域自动降权。",
                )
                inference_precision = gr.Dropdown(
                    choices=[
                        ("FP32（默认，结果基线）", "fp32"),
                        ("FP16 复原 + FP32 检测（轻量化实验）", "fp16"),
                    ],
                    value="fp32",
                    label="推理数值精度",
                    info="FP16 仅用于 CUDA 上的复原网络；安全检测器保持 FP32，正式启用前仍须比较输出差异。",
                )
                perception_guard = gr.Checkbox(
                    value=True,
                    label="启用检测安全门控",
                    info="按时序候选→逐帧候选→输入帧分层选择；只有保留道路目标证据的候选才会输出。",
                )
                guard_interval = gr.Dropdown(
                    choices=[
                        ("风险自适应（实验消融）", "adaptive"),
                        ("每帧评估（安全优先）", 1),
                        ("每2帧评估（平衡）", 2),
                        ("每3帧评估（轻量）", 3),
                        ("每5帧评估（速度优先）", 5),
                    ],
                    value=1,
                    label="检测门控评估间隔",
                    info="默认逐帧最安全；实验自适应模式会在高余量时最多复用 2 帧，须结合风险漏判统计评估。",
                )
                video_button = gr.Button(
                    "开始视频仿真",
                    variant="primary",
                    elem_classes=["sr-action"],
                )
            with gr.Column(scale=7, elem_classes=["sr-card"]):
                video_output = gr.Video(
                    label="保真融合 + 道路目标检测结果",
                    format="mp4",
                    height=390,
                    elem_classes=["sr-main-output"],
                )
                video_stats = gr.Markdown(
                    """
                    <div class="sr-note">
                    <b>等待视频</b><br>
                    建议先用默认 30 帧、640 最大边测试。显卡越快，实测处理 FPS 越高。
                    </div>
                    """
                )
                safety_timeline = gr.Image(
                    label="逐帧安全决策时间轴",
                    interactive=False,
                    height=300,
                )
                trace_files = gr.File(
                    label="完整结果包 ZIP / 逐帧 JSON / CSV / 时间轴 PNG",
                    file_count="multiple",
                    interactive=False,
                )

        with gr.Accordion("最近视频实验对比", open=True):
            gr.Markdown(
                "表格直接读取每次运行保存的 JSON，不依赖手工填写；可用于快速比较分辨率、速度、"
                "时序稳定收益和检测安全门控决策。"
            )
            history_table = gr.Dataframe(
                headers=VIDEO_HISTORY_HEADERS,
                value=load_video_history(),
                interactive=False,
                wrap=True,
                label="最近 10 次有效实验",
            )
            refresh_history = gr.Button("刷新实验历史", size="sm")

        with gr.Accordion("真实数据科研证据", open=True):
            guard_evidence = gr.Markdown(load_guard_evidence())
            performance_evidence = gr.Markdown(load_performance_evidence())
            precision_evidence = gr.Markdown(load_precision_evidence())
            refresh_evidence = gr.Button("刷新 ACDC 评测证据", size="sm")

        gr.HTML(
            f"""
            <div class="sr-note">
              <b>研究原型说明：</b>当前使用{MODEL_LABEL}与验证集冻结的融合参数。
              权重主要来自工程合成数据；正式ACDC结果用于揭示复原对检测的影响，不应把单张演示视为科研结论。
            </div>
            <div class="sr-footer">SafeRestore-RF · 本地运行 · 图片不会上传到外部服务</div>
            """
        )
        image_event = run_button.click(
            fn=models.process,
            inputs=input_image,
            outputs=[
                restored,
                conservative,
                uncertainty,
                change_heatmap,
                dcp,
                detected,
                stats,
                image_bundle,
            ],
        )
        image_event.then(fn=load_image_history, outputs=image_history_table)
        refresh_image_history.click(fn=load_image_history, outputs=image_history_table)
        batch_event = batch_button.click(
            fn=models.process_batch,
            inputs=batch_input,
            outputs=[batch_table, batch_stats, batch_files],
        )
        batch_event.then(fn=load_image_history, outputs=image_history_table)
        video_event = video_button.click(
            fn=models.process_video,
            inputs=[
                video_input,
                max_frames,
                output_fps,
                max_side,
                temporal_strength,
                perception_guard,
                guard_interval,
                inference_precision,
            ],
            outputs=[video_output, video_stats, safety_timeline, trace_files],
        )
        video_event.then(fn=load_video_history, outputs=history_table)
        refresh_history.click(fn=load_video_history, outputs=history_table)
        refresh_evidence.click(
            fn=load_evidence_bundle,
            outputs=[guard_evidence, performance_evidence, precision_evidence],
        )
    return interface.queue(default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load models and process the bundled sample, then exit.",
    )
    parser.add_argument(
        "--check-video",
        action="store_true",
        help="Process two frames of the bundled ACDC video, then exit.",
    )
    parser.add_argument(
        "--check-batch",
        action="store_true",
        help="Process the bundled fog and rain images as one batch, then exit.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local server without opening a browser window.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Inference precision used by --check-video.",
    )
    args = parser.parse_args()
    configure_local_network()
    models = DemoModels()
    models.warm_up()
    if args.check:
        sample_path = ROOT / "runs" / "assets" / "fog.jpg"
        with Image.open(sample_path) as image:
            results = models.process(np.asarray(image.convert("RGB")))
        print(f"Demo check passed on {models.device}; outputs={len(results)}")
        return
    if args.check_video:
        if not DEMO_VIDEO_PATH.is_file():
            raise FileNotFoundError(
                f"Bundled video is missing: {DEMO_VIDEO_PATH}. "
                "Run scripts/build_acdc_demo_video.py first."
            )
        output, stats, timeline, traces = models.process_video(
            str(DEMO_VIDEO_PATH),
            2,
            10.0,
            640,
            0.18,
            True,
            1,
            args.precision,
        )
        print(
            f"Video check passed on {models.device}; output={output}; "
            f"timeline={timeline}; traces={traces}\n{stats}"
        )
        return
    if args.check_batch:
        sample_paths = [ROOT / "runs" / "assets" / name for name in ("fog.jpg", "rain.jpg")]
        missing = [str(path) for path in sample_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Bundled batch samples are missing: {missing}")
        rows, stats, files = models.process_batch(
            [str(path) for path in sample_paths],
            progress=lambda *_args, **_kwargs: None,
        )
        print(f"Batch check passed on {models.device}; rows={len(rows)}; files={files}\n{stats}")
        return
    interface = build_interface(models)
    # Let Gradio select the first available local port, starting from 7860.
    # This avoids a startup failure when a previous demo process still owns 7860.
    interface.launch(
        server_name="127.0.0.1",
        inbrowser=not args.no_browser,
        share=False,
        theme=APP_THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
