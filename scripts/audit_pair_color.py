"""用本地 PNG 与历史 JPEG 的逐字节重编码核验配对数据颜色契约。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replay_weather_inputs():
    """复演原有随机流，核对每种天气前三个历史输入（包含跨fog/rain的随机状态）。"""
    from scripts import build_real_weather_pairs as weather
    calib = json.loads((ROOT / "data/real_weather/calibration.json").read_text())
    rng = random.Random(42)
    checks = []
    convert = weather.calibration_rgb
    try:
        weather.calibration_rgb = lambda entry, key: np.asarray(entry[key], dtype=np.float32)
        for name in ("fog", "rain"):
            checked = 0
            for path in sorted((ROOT / "data/raw/ACDC_images/rgb_anon" / name / "train_ref").rglob("*.png")):
                clean = cv2.resize(read(path), (960, 540), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, ::-1] / 255
                stem = path.stem.replace("_rgb_ref_anon", "")
                for k in range(3):
                    entry = rng.choice(calib[name])
                    synth = weather.synth_fog if name == "fog" else weather.synth_rain
                    image = synth(clean, entry, rng)
                    if checked < 3:
                        old_path = ROOT / "data/real_weather/degraded/train" / f"{path.parent.name}__{stem}__{name}{k}.jpg"
                        checks.append({"historical_input": str(old_path.relative_to(ROOT)), "exact_legacy_bgr_calibration_replay": old_path.read_bytes() == encode((np.clip(image,0,1)[:, :, ::-1]*255).astype(np.uint8),95)})
                        checked += 1
                if name == "rain" and checked >= 3:
                    break
    finally:
        weather.calibration_rgb = convert
    output=ROOT/"results/next_phase/color_audit.json"
    result=json.loads(output.read_text(encoding="utf-8"))
    result["weather_input_replay"] = checks
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


def read(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def encode(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def verify_fixed_versions():
    output=ROOT/"results/next_phase/color_audit.json"
    result=json.loads(output.read_text(encoding="utf-8"))
    validation={}
    for kind, split in (("real_scene", "train"), ("real_weather", "train"), ("real_weather", "dev")):
        manifest=ROOT/"data"/f"{kind}_rgb_v2"/"manifests"/f"{split}.csv"
        old_manifest=ROOT/"data"/kind/"manifests"/f"{split}.csv"
        rows=list(csv.DictReader(manifest.open(encoding="utf-8")))
        previous=list(csv.DictReader(old_manifest.open(encoding="utf-8")))
        assert len(rows)==len(previous)
        assert all(all(a[k]==b[k] for k in ("scene_id","weather","sequence")) for a,b in zip(rows,previous))
        paths=sorted({r[k] for r in rows for k in ("clean_path","degraded_path")})
        stream=hashlib.sha256()
        sizes=Counter()
        for name in paths:
            image=read(ROOT/name)
            assert image is not None, name
            sizes[f"{image.shape[1]}x{image.shape[0]}x{image.shape[2]}"]+=1
            stream.update(name.replace("\\", "/").encode())
            stream.update(hashlib.sha256((ROOT/name).read_bytes()).digest())
        corrected_targets=0
        seen=set()
        for row,old in zip(rows,previous):
            if row["clean_path"] in seen:
                continue
            seen.add(row["clean_path"])
            target=ROOT/row["clean_path"]
            if kind=="real_weather":
                corrected_targets+=int(target.read_bytes()==(ROOT/old["clean_path"]).read_bytes())
            else:
                weather=row["weather"].replace("_selfsup","")
                stem=row["scene_id"].split("__")[0]
                source=ROOT/"data/raw/ACDC_images/rgb_anon"/weather/"train"/row["sequence"]/f"{stem}_rgb_anon.png"
                full=read(source)
                first=cv2.imdecode(np.frombuffer(encode(full,95),np.uint8),cv2.IMREAD_COLOR)
                expected=encode(cv2.resize(first,(960,540),interpolation=cv2.INTER_AREA),95)
                corrected_targets+=int(target.read_bytes()==expected)
        assert corrected_targets==len(seen), (kind,split,corrected_targets,len(seen))
        assert sizes==Counter({"960x540x3":len(paths)}), sizes
        entry={"manifest":str(manifest.relative_to(ROOT)),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"pairs":len(rows),"sample_ids_and_order_unchanged":True,"decoded_files":len(paths),"invalid_files":0,"image_shapes":dict(sizes),"unique_targets_correct":corrected_targets,"degraded_files_changed":sum((ROOT/a["degraded_path"]).read_bytes()!=(ROOT/b["degraded_path"]).read_bytes() for a,b in zip(rows,previous)),"files_sha256_stream":stream.hexdigest()}
        validation[f"{kind}/{split}"]=entry
        print(json.dumps(entry,ensure_ascii=False),flush=True)
    result["fixed_version_validation"]=validation
    result["historical_manifests_unchanged"]=all(hashlib.sha256((ROOT/d["manifest"]).read_bytes()).hexdigest()==d["manifest_sha256"] for d in result["datasets"].values())
    result["historical_calibration_unchanged"]=hashlib.sha256((ROOT/result["calibration"]["path"]).read_bytes()).hexdigest()==result["calibration"]["sha256"]
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    result = {"protocol": "本地原始PNG→指定JPEG质量重编码，与历史产物逐字节比较", "datasets": {}}
    for kind in ("real_scene", "real_weather"):
        manifest = ROOT / "data" / kind / "manifests/train.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
        unique = {r["clean_path"]: r for r in rows}
        counts = Counter()
        examples = []
        for i, row in enumerate(unique.values()):
            weather = row["weather"].replace("_selfsup", "")
            if kind == "real_scene":
                stem = row["scene_id"].split("__")[0]
                source = ROOT / "data/raw/ACDC_images/rgb_anon" / weather / "train" / row["sequence"] / f"{stem}_rgb_anon.png"
                bgr = read(source)
                correct = bgr
                swapped = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                quality = 95
            else:
                stem = Path(row["clean_path"]).stem.split("__", 1)[1]
                source = ROOT / "data/raw/ACDC_images/rgb_anon" / weather / "train_ref" / row["sequence"] / f"{stem}_rgb_ref_anon.png"
                bgr = read(source)
                correct = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
                swapped = correct[:, :, ::-1]
                quality = 97
            assert bgr is not None, source
            saved = (ROOT / row["clean_path"]).read_bytes()
            saved_bgr = read(ROOT / row["clean_path"])
            if kind == "real_scene" and saved_bgr.shape != bgr.shape:
                # 历史 downscale_selfsup.py 先读回已编码 JPEG，再缩至960×540并二次编码。
                size = (saved_bgr.shape[1], saved_bgr.shape[0])
                correct = cv2.resize(cv2.imdecode(np.frombuffer(encode(correct, quality), np.uint8), cv2.IMREAD_COLOR), size, interpolation=cv2.INTER_AREA)
                swapped = cv2.resize(cv2.imdecode(np.frombuffer(encode(swapped, quality), np.uint8), cv2.IMREAD_COLOR), size, interpolation=cv2.INTER_AREA)
            ok_correct = saved == encode(correct, quality)
            ok_swapped = saved == encode(swapped, quality)
            counts["exact_correct_color"] += int(ok_correct)
            counts["exact_channel_swapped"] += int(ok_swapped)
            counts["neither"] += int(not ok_correct and not ok_swapped)
            if len(examples) < 2 or (weather in ("snow", "rain") and len(examples) == 2):
                saved_rgb = cv2.cvtColor(read(ROOT / row["clean_path"]), cv2.COLOR_BGR2RGB).astype(np.float32)
                expected_rgb = correct[:, :, ::-1].astype(np.float32)
                examples.append({"source": str(source.relative_to(ROOT)), "historical_target": row["clean_path"], "exact_correct_color": ok_correct, "exact_channel_swapped": ok_swapped, "rgb_mae_vs_source": float(np.abs(saved_rgb-expected_rgb).mean()), "rgb_mae_after_channel_reversal": float(np.abs(saved_rgb[:, :, ::-1]-expected_rgb).mean()), "historical_sha256": hashlib.sha256(saved).hexdigest()})
            if (i + 1) % 100 == 0:
                print(kind, i + 1, dict(counts), flush=True)
        result["datasets"][kind] = {"manifest": str(manifest.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "pairs": len(rows), "weather_pairs": dict(Counter(r["weather"] for r in rows)), "unique_targets": len(unique), "existing_inputs": sum((ROOT/r["degraded_path"]).is_file() for r in rows), "target_encoding": dict(counts), "examples": examples}
    calib_path = ROOT / "data/real_weather/calibration.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    checks = []
    for weather in ("fog", "rain"):
        source = sorted((ROOT / "data/raw/ACDC_images/rgb_anon" / weather / "train").rglob("*.png"))[0]
        small = cv2.resize(read(source), (480,270), interpolation=cv2.INTER_AREA).astype(np.float32)/255
        if weather == "rain":
            flat=small.reshape(-1,3)
            actual = {"mean_c": np.round(flat.mean(0),4).tolist(), "std_c": np.round(flat.std(0)+1e-6,4).tolist()}
        else:
            dark = small.min(2).ravel()
            k=max(1,int(.001*dark.size))
            air=np.clip(small.reshape(-1,3)[np.argpartition(dark,-k)[-k:]].mean(0),.45,1)
            actual={"air":np.round(air,4).tolist()}
        checks.append({"weather":weather,"source":str(source.relative_to(ROOT)),"stored":{k:calib[weather][0][k] for k in actual},"recomputed_bgr":actual,"matches_bgr":all(np.allclose(v,calib[weather][0][k],atol=1e-5) for k,v in actual.items())})
    result["calibration"]={"path":str(calib_path.relative_to(ROOT)),"sha256":hashlib.sha256(calib_path.read_bytes()).hexdigest(),"checks":checks}
    versions=[]
    for config_path in sorted((ROOT/"runs").glob("*/config.json")):
        config=json.loads(config_path.read_text(encoding="utf-8"))
        train=str(config.get("train_manifest",""))
        extras=config.get("extra_manifest",[])
        if "real_weather/" in train or any("real_scene/" in p for p in extras):
            versions.append({"name":config.get("name",config_path.parent.name),"train_manifest":train,"extra_manifest":extras,"weather_uses_at_risk_manifest":"real_weather/" in train,"scene_swapped_targets":any("real_scene/" in p for p in extras)})
    result["configured_versions"]=versions
    out=ROOT/"results/next_phase/color_audit.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(out,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-inputs", action="store_true")
    parser.add_argument("--verify-fixed", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.replay_inputs:
        replay_weather_inputs()
    elif args.verify_fixed:
        verify_fixed_versions()
    else:
        main()
