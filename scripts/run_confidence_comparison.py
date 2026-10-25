"""离线顺序运行同预算v4续训对照与直接置信度保护试验。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--confidence-weight", type=float, default=10.0)
    parser.add_argument("--name", default="confidence_v8_20261025")
    args = parser.parse_args()
    if not 0 < args.steps <= 16000 or not math.isfinite(args.confidence_weight) or args.confidence_weight <= 0:
        parser.error("步数必须在1至16000之间，实验组置信度权重必须有限且大于0")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("实验名只能使用英文字母、数字、下划线或短横线")
    folder = ROOT / "runs" / args.name
    if folder.exists():
        raise FileExistsError(f"拒绝覆盖已有实验：{folder}")
    arm_names = [args.name + "_control", args.name + "_guard"]
    for name in arm_names:
        if (ROOT / "runs" / name).exists():
            raise FileExistsError(f"拒绝覆盖已有训练目录：{name}")
    required = ["runs/real_v4_allweather/latest.pt", "yolov8n.pt",
                "data/real_weather/manifests/train.csv", "data/real_scene/manifests/train.csv",
                "runs/acdc_benchmark/evaluation/weather_input_all406.json"]
    if any(not (ROOT / p).is_file() for p in required):
        raise FileNotFoundError("本地实验资源不完整，禁止联网补齐")
    folder.mkdir(parents=True)
    sources = ["scripts/train_restorer.py", "src/saferestore/losses/perceptual.py",
               "src/saferestore/losses/confidence.py", "src/saferestore/models/nafnet.py",
               "scripts/run_confidence_comparison.py", "scripts/diagnose_detection_rootcause.py",
               "scripts/evaluate_coco_detection.py", "configs/detector.yaml"]
    plan = {"steps_per_arm": args.steps, "total_lr_schedule": 64000,
            "confidence_weight": args.confidence_weight, "arms": arm_names,
            "inputs_sha256": {p: sha256(ROOT / p) for p in required},
            "source_sha256": {p: sha256(ROOT / p) for p in sources},
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "ultralytics", "numpy")},
            "scope": "ACDC val模型选择；非独立测试；单seed42续训",
            "status": "running"}
    (folder / "comparison_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    common = ["--variant", "standard", "--crop", "320", "--batch", "4", "--iters", "64000",
              "--stop-after", str(args.steps), "--resume", required[0],
              "--extra-manifest", required[3], "--extra-ratio", "0.6",
              "--w-det-feat", "2.0", "--det-feat-start", "0", "--det-feat-crop", "256",
              "--det-feat-view", "crop", "--det-feat-scales", "2,1,0.75",
              "--det-feat-magnitude-weight", "0", "--edge-px", "3.0", "--w-delta", "0.05",
              "--det-conf-monitor", "--det-conf-threshold", "0.25", "--det-conf-topk", "16",
              "--eval-det-every", str(args.steps), "--eval-records", required[4],
              "--det-limit", "406", "--save-every", "2000", "--log-every", "100", "--seed", "42"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": "src;.", "YOLO_OFFLINE": "1"}
    for name, weight in zip(arm_names, (0.0, args.confidence_weight)):
        cmd = [sys.executable, "-u", "scripts/train_restorer.py", "--name", name,
               *common, "--w-det-conf", str(weight)]
        print(f"启动 {name}，权重={weight}，新增{args.steps}步", flush=True)
        with (folder / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            plan["status"] = "failed"
            plan["failed_arm"] = name
            (folder / "comparison_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            return code
        if any(sha256(ROOT / p) != h for p, h in plan["inputs_sha256"].items()):
            raise RuntimeError("源资源哈希变化，停止公平对照")
        if any(sha256(ROOT / p) != h for p, h in plan["source_sha256"].items()):
            raise RuntimeError("训练源码或检测协议变化，停止公平对照")
    configs = [json.loads((ROOT / "runs" / n / "config.json").read_text(encoding="utf-8")) for n in arm_names]
    for key in ("name", "w_det_conf"):
        for config in configs:
            config.pop(key)
    assert configs[0] == configs[1], "两组除名称/置信度权重外配置不一致"
    traces = [json.loads((ROOT / "runs" / n / "batch_trace.json").read_text()) for n in arm_names]
    assert traces[0] == traces[1], "两组完整输入/目标批次不一致"
    monitors = [json.loads((ROOT / "runs" / n / "confidence_monitor.json").read_text(encoding="utf-8")) for n in arm_names]
    assert monitors[0]["selected_anchors"] == monitors[1]["selected_anchors"], "教师监督位置不一致"
    plan.update(status="completed", config_match=True, batch_match=True, teacher_selection_match=True)
    (folder / "comparison_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
