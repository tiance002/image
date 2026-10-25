"""在两组续训结束后，离线生成以输入原图为基线的完整置信度保护对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}


def read_json(path: Path):
    """读取本地 UTF-8 证据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    """不依赖可被优化选项关闭的 assert 验证证据。"""
    if not condition:
        raise ValueError(message)


def validate_tables(records: dict, tables: dict[str, dict]) -> dict[str, list[str]]:
    """严格检查完整图像及逐图真值，禁止静默取交集。"""
    require(len(records) == 406, "必须恰好包含406张记录")
    scopes = {"all406": list(records)}
    for weather in ("fog", "rain", "night", "snow"):
        scopes[weather] = [key for key, row in records.items() if row["weather"] == weather]
        require(bool(scopes[weather]), f"缺少天气：{weather}")
    require(len(scopes["fog"]) == 100 and len(scopes["rain"]) == 100,
            "主口径必须为fog100张与rain100张")
    require(sum(len(scopes[w]) for w in ("fog", "rain", "night", "snow")) == 406,
            "记录包含未支持的天气")
    scopes["rain_fog200"] = scopes["fog"] + scopes["rain"]
    for name, table in tables.items():
        require(set(table) == set(records), f"{name}图像集合与406张记录不完全一致")
        for key, row in records.items():
            require(table[key]["ground_truth"] == row["ground_truth"],
                    f"{name}真值不一致：{key}")
            require(set(row["ground_truth"]).issubset({str(c) for c in CLASSES}),
                    f"记录存在不支持的GT类别：{key}")
    return scopes


def validate_completed(folder: Path) -> tuple[dict, list[str], dict]:
    """完成标记之外，再核对实际配置、批次轨迹和教师选择。"""
    plan = read_json(folder / "comparison_plan.json")
    require(plan.get("status") == "completed", "比较尚未完成，拒绝生成最终结果")
    for field in ("config_match", "batch_match", "teacher_selection_match"):
        require(plan.get(field) is True, f"缺少公平性通过标记：{field}")
    require(plan.get("steps_per_arm") == 4000 and plan.get("total_lr_schedule") == 64000,
            "本报告固定为各4000步、64000步学习率日程")
    require(plan.get("confidence_weight") == 10.0, "本报告要求预设置信度权重10")
    arms = plan.get("arms", [])
    require(len(arms) == 2 and len(set(arms)) == 2, "需要两个不同的control/guard实验目录")
    require(all(isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in arms),
            "实验目录名称无效")
    configs = [read_json(ROOT / "runs" / name / "config.json") for name in arms]
    require([c.get("w_det_conf") for c in configs] == [0.0, 10.0],
            "对照/保护组权重必须依次为0和10")
    require(all(c.get("det_conf_monitor") is True for c in configs), "两组都必须开启教师监测")
    require(all(c.get("det_conf_threshold") == 0.25 and c.get("det_conf_topk") == 16
                for c in configs), "教师阈值/topk必须为预设的.25/16")
    require(all(c.get("seed") == 42 and c.get("stop_after") == 4000 and c.get("iters") == 64000
                for c in configs), "实际训练必须为seed42、4000步与64000学习率日程")
    source_checkpoint = (ROOT / "runs/real_v4_allweather/latest.pt").resolve()
    require(all((ROOT / c.get("resume", "")).resolve() == source_checkpoint for c in configs),
            "两组必须从固定历史v4/latest检查点恢复")
    normalized = [{k: v for k, v in c.items() if k not in {"name", "w_det_conf"}}
                  for c in configs]
    require(normalized[0] == normalized[1], "实际配置除名称/新增权重外不一致")
    traces = [read_json(ROOT / "runs" / n / "batch_trace.json") for n in arms]
    require(traces[0] == traces[1], "完整输入/目标批次轨迹不一致")
    require(traces[0].get("batches") == 4000, "批次哈希必须覆盖全部4000步")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", traces[0].get("sha256", ""))), "批次哈希格式无效")
    monitors = [read_json(ROOT / "runs" / n / "confidence_monitor.json") for n in arms]
    require(all(m.get("iter") == 51999 for m in monitors), "教师监测必须覆盖最终iter51999")
    for field in ("selected_anchors", "supervised_image_class_pairs"):
        for monitor in monitors:
            values = monitor.get(field, {})
            require(set(values) == set(CLASSES.values()), f"教师监测缺少完整七类：{field}")
            require(all(type(v) is int and v >= 0 for v in values.values()),
                    f"教师监测计数无效：{field}")
        require(monitors[0][field] == monitors[1][field], f"两组教师监测不一致：{field}")
    # 检查计划中冻结的资源与源代码，报告本身不修改其中任何文件。
    for group in ("inputs_sha256", "source_sha256"):
        for relative, expected in plan[group].items():
            digest = hashlib.sha256()
            with (ROOT / relative).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            require(digest.hexdigest() == expected, f"冻结证据哈希变化：{relative}")
    return plan, arms, dict(zip(("control", "guard"), monitors))


def delta(value, baseline):
    """缺少真值导致无定义的指标保留为空值。"""
    return None if value is None or baseline is None else value - baseline


def build_summary(folder: Path) -> dict:
    """只使用已经完成的本地预测，不启动推理或训练。"""
    plan, arms, monitors = validate_completed(folder)
    # 延后导入既有指标模块，帮助命令无需加载检测器依赖。
    from scripts.report_d0 import compare, load_table
    paths = {
        "input": "runs/acdc_benchmark/evaluation/weather_input_direct_406.json",
        "v4_48k": "runs/real_v4_allweather/preds_047999.json",
        "control_52k": f"runs/{arms[0]}/preds_051999.json",
        "guard_52k": f"runs/{arms[1]}/preds_051999.json",
    }
    records_path = "runs/acdc_benchmark/evaluation/weather_input_all406.json"
    records = load_table(ROOT / records_path)
    tables = {name: load_table(ROOT / path) for name, path in paths.items()}
    scopes = validate_tables(records, tables)
    results = {}
    for scope, keys in scopes.items():
        # 每一项OPR与保留率均相对真正的输入原图，绝不换成control。
        comparisons = {name: compare(tables["input"], table, keys)
                       for name, table in tables.items()}
        control, guard = (comparisons[n] for n in ("control_52k", "guard_52k"))
        difference = {metric: delta(guard["method"][metric], control["method"][metric])
                      for metric in ("map50", "map50_without_train", "recall", "precision",
                                     "false_positives_per_image")}
        difference.update(
            opr_relative_input=delta(guard["opr"], control["opr"]),
            retention_relative_input=delta(guard["retention"], control["retention"]),
            per_class_ap50={name: delta(guard["method"]["per_class"][name]["ap50"],
                                       control["method"]["per_class"][name]["ap50"])
                            for name in CLASSES.values()},
        )
        for row in comparisons.values():
            row["delta_map50_vs_input"] = delta(row["method"]["map50"], row["baseline"]["map50"])
            row["weather_map_redline_pass"] = row["delta_map50_vs_input"] >= -0.03
        results[scope] = {"vs_input": comparisons, "guard_minus_control": difference}
    gates = {}
    for name in ("v4_48k", "control_52k", "guard_52k"):
        primary = results["rain_fog200"]["vs_input"][name]
        checks = {
            "retention_ge_95pct": primary["retention"] is not None and primary["retention"] >= .95,
            "opr_ge_85pct": primary["opr"] is not None and primary["opr"] >= .85,
            "class_redline_pass": primary["class_redline_pass"],
            "recall_redline_pass": primary["recall_redline_pass"],
            "weather_redline_pass": all(results[w]["vs_input"][name]["weather_map_redline_pass"]
                                         for w in ("fog", "rain", "night", "snow")),
        }
        gates[name] = {**checks, "detection_only_val_pass": all(checks.values()),
                       "full_d1_verified": False}
    return {
        "scope": "ACDC val模型选择；非独立test；同一v4检查点seed42续训",
        "comparison": folder.name, "sources": {**paths, "records": records_path},
        "fairness_plan": plan, "teacher_monitor": monitors,
        "teacher_monitor_note": "教师预测计数，不是GT、独立图像数或真实正样本覆盖率",
        "scopes": results, "detection_only_val_checks_vs_input": gates,
        "limitations": ["未验证独立test、多种子、统计显著性或PSNR/LPIPS红线",
                        "置信度约束不包含框回归，不保证定位或NMS后的框保留"],
    }


def number(value) -> str:
    """Markdown中保留无定义指标，不将其写成零。"""
    return "—" if value is None else f"{value:.6f}"


def render_report(output: dict) -> str:
    """完整列出各口径D0和相对同预算对照的变化。"""
    names = ("input", "v4_48k", "control_52k", "guard_52k")
    lines = ["# 置信度保护同预算对照（离线完整结果）", "", output["scope"], "",
             "所有OPR、保留率、类别红线和天气差值均相对输入原图；control仅用于新增损失的增量比较。",
             "仅检查检测指标的val筛选规则，完整D1未验证；无独立test、多种子或显著性结论。",
             "PSNR/LPIPS红线未在本报告验证。", "", "## 来源与公平性", "",
             "计划完成标记、配置、全部批次轨迹、两类教师计数与冻结文件哈希均已核对。", ""]
    lines += [f"- {name}: `{path}`" for name, path in output["sources"].items()]
    for scope in ("rain_fog200", "all406", "fog", "rain", "night", "snow"):
        rows = output["scopes"][scope]["vs_input"]
        lines += ["", f"## {scope}（{rows['input']['method']['n_images']}张）", "",
                  "| 指标 | input | v4@48k | control@52k | guard@52k |",
                  "|---|---:|---:|---:|---:|"]
        for metric in ("map50", "map50_without_train", "recall", "precision", "false_positives_per_image"):
            lines.append(f"| {metric} | " + " | ".join(number(rows[n]["method"][metric]) for n in names) + " |")
        for metric in ("retention", "opr", "delta_map50_vs_input"):
            lines.append(f"| {metric}（相对输入） | " + " | ".join(number(rows[n][metric]) for n in names) + " |")
        lines += ["", "| 类别 | GT | 输入AP50 | v4 AP50 | control AP50 | guard AP50 | guard−control |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for class_name in CLASSES.values():
            metrics = [rows[n]["method"]["per_class"][class_name] for n in names]
            lines.append(f"| {class_name} | {metrics[0]['ground_truth']} | "
                         + " | ".join(number(m["ap50"]) for m in metrics) + " | "
                         + number(delta(metrics[3]["ap50"], metrics[2]["ap50"])) + " |")
        lines += ["", "| 类别 | v4丢失/新增 | control丢失/新增 | guard丢失/新增 |",
                  "|---|---:|---:|---:|"]
        for class_name in CLASSES.values():
            values = [rows[n]["target_changes"][class_name] for n in names[1:]]
            lines.append(f"| {class_name} | " + " | ".join(f"{v['lost']}/{v['gained']}" for v in values) + " |")
        lines += ["", "丢失/新增均以输入已正确检出GT集合为参照，采用一对一IoU≥.5匹配。", "",
                  "guard−control汇总（OPR差仍是两者各自相对输入的OPR之差）：", ""]
        change = output["scopes"][scope]["guard_minus_control"]
        lines += [f"- {k}: {number(v)}" for k, v in change.items() if k != "per_class_ap50"]
        if scope in {"fog", "rain", "night", "snow"}:
            failures = [n for n in names[1:] if not rows[n]["weather_map_redline_pass"]]
            lines += ["", "相对输入的天气mAP下降超过3个百分点：" + ("、".join(failures) or "无")]
        if rows["input"]["method"]["per_class"]["motorcycle"]["ap50"] == 0:
            lines += ["", "motorcycle 在基线下的 AP 为 0.0000，属检测器能力上限而非复原引入的退化。"]
    lines += ["", "## 相对输入的检测指标val检查", "",
              "该表不等于完整D1验收。", "", "```json",
              json.dumps(output["detection_only_val_checks_vs_input"], ensure_ascii=False, indent=2),
              "```", "", "## 教师预测监督覆盖", "", output["teacher_monitor_note"], "",
              "| 类别 | control锚点 | guard锚点 | control图像/类别对 | guard图像/类别对 |",
              "|---|---:|---:|---:|---:|"]
    monitors = output["teacher_monitor"]
    for name in CLASSES.values():
        values = [monitors[arm][field][name] for field in
                  ("selected_anchors", "supervised_image_class_pairs") for arm in ("control", "guard")]
        lines.append(f"| {name} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "bus/train教师覆盖为零或很少时，应报告机制覆盖不足；不能解释为已充分检验。",
              "无直接定位约束，false positives也必须与AP、OPR一起判读。",
              "本报告不自动部署模型、不调整参数、不替换v4。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="confidence_v8_20261025", help="runs下的比较目录名称")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", args.comparison)), "比较目录名称无效")
    folder = ROOT / "runs" / args.comparison
    output = build_summary(folder)
    (folder / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "REPORT.md").write_text(render_report(output), encoding="utf-8")
    print(folder / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
