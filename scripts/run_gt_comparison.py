"""真实GT检测监督的预注册、梯度校准及C0/C1同预算顺序训练；不自动评测或发布。"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_restorer import (PairDataset, charbonnier, color_constancy,
                                    edge_weight_map, gradient_loss, ssim_index)
from saferestore.data.detection_gt import ACDCDetectionDataset, collate_detection_batch
from saferestore.losses.perceptual import DetectorFeatureLoss, prepare_detector_view
from saferestore.losses.task_detection import FrozenYoloTaskLoss
from saferestore.models.nafnet import build_nafnet


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def state_sha256(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_datasets(config):
    paired = PairDataset(ROOT / config["train_manifest"], config["crop"],
                         length=max(config["steps"], config["calibration_batches"]) * config["batch"],
                         seed=config["seed"], min_scale=config["scale_jitter"],
                         extra_manifests=[ROOT / config["extra_manifest"]], extra_ratio=config["extra_ratio"])
    detection = ACDCDetectionDataset(ROOT / config["gt_annotations"], ROOT / config["gt_images_root"],
                                     image_size=config["gt_image_size"], horizontal_flip=config["gt_flip"], seed=config["seed"])
    validation = json.loads((ROOT / config["validation_annotations"]).read_text(encoding="utf-8"))
    train_names = {record["file_name"] for record in detection.records}
    val_names = {str(image["file_name"]).replace("\\", "/") for image in validation["images"]}
    if train_names & val_names or {Path(name).name for name in train_names} & {Path(name).name for name in val_names}:
        raise ValueError("train/val天气帧重复，禁止训练")
    return paired, detection


def input_manifest(config, detection):
    paths = [config[key] for key in ("resume", "v4_config", "detector_weights", "train_manifest", "extra_manifest", "gt_annotations", "validation_annotations")]
    # 每一张配对训练图和真实天气帧都固定内容哈希，防止文件名相同但内容变更。
    for manifest in (config["train_manifest"], config["extra_manifest"]):
        with (ROOT / manifest).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                paths.extend([row["degraded_path"], row["clean_path"]])
    paths.extend(str(record["path"].relative_to(ROOT)) for record in detection.records)
    result = {}
    for name in sorted(set(paths)):
        result[name.replace("\\", "/")] = sha256(ROOT / name)
    return result


def prepare(config, run_root):
    paired, detection = build_datasets(config)
    source_paths = ["scripts/run_gt_comparison.py", "scripts/train_restorer.py", "src/saferestore/data/detection_gt.py",
                    "src/saferestore/losses/task_detection.py", "src/saferestore/losses/perceptual.py", "src/saferestore/models/nafnet.py"]
    import ultralytics
    expected = {"config": config, "inputs_sha256": input_manifest(config, detection),
                "source_sha256": {name: sha256(ROOT / name) for name in source_paths},
                "packages": {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "numpy": np.__version__},
                "gt_counts": detection.counts, "gt_images": len(detection),
                "data_distribution_note": "v4使用历史数据：scene800通道交换已实证；weather历史输入与当前生成器seed42回放不一致，不能断言历史weather全部受当前代码缺陷影响。两组共用新生成RGBv2数据；weather新旧差异可能不止通道，不能归因为纯颜色修复。官方train/val无图像交集，但有15个weather/sequence键重合。"}
    plan_path = run_root / "comparison_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in expected:
            if previous.get(key) != expected[key]:
                raise ValueError(f"预注册后资源/配置改变：{key}；请使用新name，不覆盖已有试验")
        plan = previous
    else:
        plan = dict(expected, status="prepared", arms=[config["name"] + "_C0", config["name"] + "_C1"])
        write_json(plan_path, plan)
    return paired, detection, plan


def initialize(config, device):
    seed_everything(config["seed"])
    checkpoint = torch.load(ROOT / config["resume"], map_location="cpu", weights_only=False)
    if int(checkpoint["iter"]) != 47999:
        raise ValueError("主初始化必须为v4@47999")
    model = build_nafnet(variant=config["variant"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    ema = build_nafnet(variant=config["variant"]).to(device).eval().requires_grad_(False)
    ema.load_state_dict(checkpoint["ema"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], betas=tuple(config["betas"]))
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group.update(lr=config["lr"], weight_decay=config["weight_decay"], betas=tuple(config["betas"]))
    del checkpoint
    feature = DetectorFeatureLoss(str(ROOT / config["detector_weights"]), layer_weights=tuple(config["detector_feature_scales"]),
                                  magnitude_weight=0, confidence_monitor=False).to(device).eval()
    task = FrozenYoloTaskLoss(ROOT / config["detector_weights"], device=device).eval()
    v4 = json.loads((ROOT / config["v4_config"]).read_text(encoding="utf-8"))
    return model, ema, optimizer, feature, task, v4


def restoration_loss(model, feature, degraded, clean, weights, config):
    prediction = model(degraded)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("复原分支输出非有限")
    pixel = charbonnier(prediction, clean, weight=edge_weight_map(clean, weights["edge_px"]))
    ssim = 1 - ssim_index(prediction, clean)
    gradient = gradient_loss(prediction, clean)
    color = color_constancy(prediction, clean)
    delta = (prediction - degraded).abs().mean()
    pred_view, clean_view = prepare_detector_view(prediction.clamp(0, 1), clean, config["detector_feature_size"], "crop")
    detector_feature = feature(pred_view, clean_view)
    loss = (weights["w_pixel"] * pixel + weights["w_ssim"] * ssim + weights["w_grad"] * gradient +
            weights["w_color"] * color + weights["w_delta"] * delta + weights["w_det_feat"] * detector_feature)
    components = dict(pixel=pixel, ssim=ssim, grad=gradient, color=color, delta=delta, detector_feature=detector_feature)
    return loss, {key: float(value.detach()) for key, value in components.items()}


def parameter_gradients(loss, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("损失非有限")
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    values = tuple(torch.zeros_like(parameter) if value is None else value for parameter, value in zip(parameters, values))
    if not torch.stack([torch.isfinite(value).all() for value in values]).all():
        raise FloatingPointError("梯度非有限")
    norm = torch.linalg.vector_norm(torch.stack([value.detach().norm(2) for value in values]))
    return values, float(norm)


def detection_order(length, count, seed):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        result.extend(rng.permutation(length).tolist())
    return result[:count]


def paired_loader(paired, config):
    return DataLoader(paired, batch_size=config["batch"], shuffle=False, num_workers=config["num_workers"],
                      drop_last=True, pin_memory=torch.cuda.is_available(), persistent_workers=config["num_workers"] > 0)


def task_gradients(model, task, batch, parameters, device):
    image = batch["img"].to(device)
    prediction = model(image)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("检测分支输出非有限")
    loss, stats = task(prediction, batch)
    gradients, norm = parameter_gradients(loss, parameters)
    if any(parameter.grad is not None for parameter in task.detector.parameters()):
        raise RuntimeError("冻结检测器意外获得参数梯度")
    return gradients, norm, stats


def calibrate(config, paired, detection, device, run_root, plan):
    path = run_root / "calibration.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["plan_fingerprint"] != hashlib.sha256(json.dumps({k: plan[k] for k in ("config", "inputs_sha256", "source_sha256")}, sort_keys=True).encode()).hexdigest():
            raise ValueError("旧校准与当前预注册不匹配")
        return result
    model, ema, optimizer, feature, task, weights = initialize(config, device)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    model_before = state_sha256(model)
    detector_before = state_sha256(task.detector)
    feature_before = state_sha256(feature.detector)
    order = detection_order(len(detection), config["calibration_batches"] * config["gt_microbatch"], config["seed"])
    records = []
    for index, (degraded, clean) in enumerate(paired_loader(paired, config)):
        if index >= config["calibration_batches"]:
            break
        loss, components = restoration_loss(model, feature, degraded.to(device), clean.to(device), weights, config)
        gradients, norm_a = parameter_gradients(loss, parameters)
        del gradients, loss
        # 清掉特征hook引用，确保A图在B前释放。
        feature._features.clear()
        indices = order[index * config["gt_microbatch"]:(index + 1) * config["gt_microbatch"]]
        batch = collate_detection_batch([detection[item] for item in indices])
        gradients, norm_b, stats = task_gradients(model, task, batch, parameters, device)
        del gradients
        if not (norm_a > 0 and norm_b > 0 and math.isfinite(norm_a + norm_b)):
            raise FloatingPointError("校准梯度必须有限且为正")
        records.append(dict(batch=index, restoration_grad_l2=norm_a, task_grad_l2=norm_b,
                            proposed_weight=config["calibration_gradient_ratio"] * norm_a / norm_b,
                            task=stats, restoration=components, gt_indices=indices))
        print(f"calibration {index + 1}/{config['calibration_batches']} gradA={norm_a:.6g} gradB={norm_b:.6g}", flush=True)
    coefficient = float(np.median([record["proposed_weight"] for record in records]))
    if len(records) != config["calibration_batches"] or coefficient <= 0 or not math.isfinite(coefficient):
        raise ValueError("校准不完整或权重非法")
    assert model_before == state_sha256(model), "校准不得更新NAFNet"
    assert detector_before == state_sha256(task.detector) and feature_before == state_sha256(feature.detector), "校准检测器/BN改变"
    result = dict(weight=coefficient, batches=len(records), records=records, no_optimizer_updates=True,
                  model_unchanged=True, detectors_unchanged=True,
                  plan_fingerprint=hashlib.sha256(json.dumps({k: plan[k] for k in ("config", "inputs_sha256", "source_sha256")}, sort_keys=True).encode()).hexdigest())
    write_json(path, result)
    feature.close()
    del model, ema, optimizer, feature, task, parameters
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def update_stream_hash(digest, degraded, clean, batch, paired, pair_indices):
    metadata = {"pair_indices": pair_indices, "pair_paths": [[paired._pick(i)[0][key] for key in ("degraded_path", "clean_path")] for i in pair_indices],
                "gt_indices": batch["indices"], "gt_file_names": batch["file_names"]}
    blocks = [json.dumps(metadata, sort_keys=True).encode()]
    for tensor in (degraded, clean, batch["img"], batch["cls"], batch["bboxes"], batch["batch_idx"], batch["ignore_boxes"], batch["ignore_batch_idx"]):
        blocks.extend([str((str(tensor.dtype), list(tensor.shape))).encode(), tensor.contiguous().numpy().tobytes()])
    for block in blocks:
        digest.update(struct.pack("<Q", len(block)))
        digest.update(block)
    return metadata


def run_arm(arm, config, paired, detection, device, coefficient, run_root):
    directory = ROOT / "runs" / f"{config['name']}_{arm}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", dict(config, arm=arm, calibrated_task_weight=coefficient,
                                               task_gradient_applied=arm == "C1"))
    model, ema, optimizer, feature, task, weights = initialize(config, device)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    frozen_before = {"native_task": state_sha256(task.detector), "feature": state_sha256(feature.detector)}
    initial_model = state_sha256(model)
    initial_ema = state_sha256(ema)
    order = detection_order(len(detection), config["steps"] * config["gt_microbatch"], config["seed"])
    digest = hashlib.sha256()
    columns = ["step", "iter", "restoration_total", "pixel", "ssim", "grad", "color", "delta", "detector_feature", "task_total", "box_loss", "cls_loss", "dfl_loss",
               "grad_restoration_l2", "grad_task_l2", "task_weight_nominal", "task_weight_applied", "grad_task_weighted_nominal_l2", "grad_task_weighted_applied_l2",
               "grad_total_preclip_l2", "grad_total_postclip_l2", "gt_count", "ignored_anchor_fraction", "lr", "elapsed_seconds", "stream_sha256"]
    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (directory / "train_log.csv").open("w", encoding="utf-8", newline="") as output, (directory / "sample_stream.jsonl").open("w", encoding="utf-8") as stream:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, (degraded, clean) in enumerate(paired_loader(paired, config)):
            if index >= config["steps"]:
                break
            indices = order[index * config["gt_microbatch"]:(index + 1) * config["gt_microbatch"]]
            batch = collate_detection_batch([detection[item] for item in indices])
            pair_indices = list(range(index * config["batch"], (index + 1) * config["batch"]))
            metadata = update_stream_hash(digest, degraded, clean, batch, paired, pair_indices)
            stream.write(json.dumps(dict(step=index + 1, **metadata, sha256=digest.hexdigest()), ensure_ascii=False) + "\n")
            optimizer.zero_grad(set_to_none=True)
            loss, components = restoration_loss(model, feature, degraded.to(device), clean.to(device), weights, config)
            loss_a_value = float(loss.detach())
            grad_a, norm_a = parameter_gradients(loss, parameters)
            del loss
            feature._features.clear()
            grad_b, norm_b, stats = task_gradients(model, task, batch, parameters, device)
            if norm_a <= 0 or norm_b <= 0:
                raise FloatingPointError("分支梯度为零，停止以防伪训练")
            nominal = coefficient * min((index + 1) / config["task_warmup_steps"], 1)
            applied = nominal if arm == "C1" else 0.0
            for parameter, ga, gb in zip(parameters, grad_a, grad_b):
                parameter.grad = ga.add_(gb, alpha=applied)
            del grad_a, grad_b
            preclip = float(torch.nn.utils.clip_grad_norm_(parameters, config["clip_grad_norm"], error_if_nonfinite=True))
            postclip = min(preclip, config["clip_grad_norm"])
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.mul_(config["ema"]).add_(parameter, alpha=1 - config["ema"])
                if not torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all():
                    raise FloatingPointError("更新后NAFNet非有限")
            done = index + 1
            row = dict(step=done, iter=47999 + done, restoration_total=loss_a_value, **components,
                       task_total=stats["total"], box_loss=stats["box_loss"], cls_loss=stats["cls_loss"], dfl_loss=stats["dfl_loss"],
                       grad_restoration_l2=norm_a, grad_task_l2=norm_b, task_weight_nominal=nominal, task_weight_applied=applied,
                       grad_task_weighted_nominal_l2=norm_b * nominal, grad_task_weighted_applied_l2=norm_b * applied,
                       grad_total_preclip_l2=preclip, grad_total_postclip_l2=postclip, gt_count=stats["gt_count"],
                       ignored_anchor_fraction=stats["ignored_anchor_fraction"], lr=config["lr"], elapsed_seconds=time.perf_counter() - start,
                       stream_sha256=digest.hexdigest())
            writer.writerow(row)
            if done % config["log_every"] == 0 or done == 1:
                output.flush()
                stream.flush()
                print(f"{arm} {done}/{config['steps']} lossA={loss_a_value:.5f} task={stats['total']:.5f} gradA={norm_a:.5g} gradB={norm_b:.5g} weight={applied:.5g} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
            if done % config["checkpoints_every"] == 0 or done == config["steps"]:
                checkpoint = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                  scaler={}, iter=47999 + done, next_phase_config=config, arm=arm, task_weight=coefficient)
                torch.save(checkpoint, directory / f"iter_{47999 + done:06d}.pt")
                torch.save(checkpoint, directory / "latest.pt")
                del checkpoint
    frozen_after = {"native_task": state_sha256(task.detector), "feature": state_sha256(feature.detector)}
    if frozen_after != frozen_before:
        raise RuntimeError("检测器/BN在训练中发生改变")
    result = dict(arm=arm, steps=done, stream_sha256=digest.hexdigest(), initial_model_sha256=initial_model,
                  initial_ema_sha256=initial_ema, frozen_before=frozen_before, frozen_after=frozen_after,
                  detector_unchanged=True, task_gradient_nonzero=True, elapsed_seconds=time.perf_counter() - start,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  checkpoint_sha256={path.name: sha256(path) for path in directory.glob("*.pt")})
    write_json(directory / "training_evidence.json", result)
    feature.close()
    del model, ema, optimizer, feature, task, parameters, ga, gb
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/next_phase_training.json")
    parser.add_argument("--name")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.name:
        config["name"] = args.name
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("steps必须为正")
        config["steps"] = args.steps
    if config["steps"] != 4000 and config["name"] == "gt_next_phase_20261025":
        parser.error("非4000步只能使用独立smoke name")
    if not config["name"].replace("_", "").replace("-", "").isalnum():
        parser.error("name仅允许字母数字下划线连字符")
    run_root = ROOT / "runs" / config["name"]
    paired, detection, plan = prepare(config, run_root)
    print(f"Plan ready: {run_root / 'comparison_plan.json'}; trainGT={len(detection)}", flush=True)
    if args.prepare_only:
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，禁止无提示更改设备或训练配置")
    if not args.calibration_only:
        for arm in ("C0", "C1"):
            if (ROOT / "runs" / f"{config['name']}_{arm}").exists():
                raise FileExistsError("试验目录已存在，拒绝覆盖，请使用新name")
    calibration = calibrate(config, paired, detection, args.device, run_root, plan)
    if args.calibration_only:
        return 0
    plan["status"] = "training"
    write_json(run_root / "comparison_plan.json", plan)
    results = [run_arm(arm, config, paired, detection, args.device, calibration["weight"], run_root) for arm in ("C0", "C1")]
    for key in ("steps", "stream_sha256", "initial_model_sha256", "initial_ema_sha256", "frozen_before"):
        if results[0][key] != results[1][key]:
            raise RuntimeError(f"公平性核验失败：{key}")
    if input_manifest(config, detection) != plan["inputs_sha256"]:
        raise RuntimeError("训练后输入资源哈希改变")
    if {name: sha256(ROOT / name) for name in plan["source_sha256"]} != plan["source_sha256"]:
        raise RuntimeError("训练期间源码改变")
    plan.update(status="completed", config_match=True, batch_match=True, initialization_match=True, detector_unchanged=True, arm_results=results)
    write_json(run_root / "comparison_plan.json", plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
