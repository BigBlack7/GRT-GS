#!/usr/bin/env python3
import argparse
import csv
import html
import os
import re
from pathlib import Path

# python data_collect.py /data2/zmh/output_physnorm_baseline_diag/step0_baseline

DEFAULT_OUTPUT_DIR = "/data2/zmh/output_physnorm_baseline_diag/step0_baseline"
METRIC_KEYS = ["psnr", "ssim", "lpips", "fps"]
DROP_THRESHOLD = 1.0


def parse_metric(metric_path):
    metrics = {}
    try:
        content = Path(metric_path).read_text(encoding="utf-8", errors="ignore")
        for key in METRIC_KEYS:
            match = re.search(rf"{key}\s*:\s*([-+0-9.eE]+)", content)
            if match:
                metrics[key] = float(match.group(1))
    except Exception as exc:
        print(f"  [警告] 读取失败: {metric_path} -> {exc}")
    return metrics


def parse_eval_curve(curve_path):
    rows = []
    path = Path(curve_path)
    if not path.exists():
        return rows
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        rows.append(
                            {
                                "iteration": int(row["iteration"]),
                                "split": row["split"],
                                "l1": None if row.get("l1", "") in {"", "N/A"} else float(row["l1"]),
                                "psnr": float(row["psnr"]),
                            }
                        )
                    except (KeyError, ValueError):
                        continue
        else:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 4 or not parts[0].isdigit():
                        continue
                    try:
                        rows.append(
                            {
                                "iteration": int(parts[0]),
                                "split": parts[1],
                                "l1": None if parts[2] == "N/A" else float(parts[2]),
                                "psnr": float(parts[3]),
                            }
                        )
                    except ValueError:
                        continue
    except Exception as exc:
        print(f"  [警告] 曲线读取失败: {curve_path} -> {exc}")
    return rows


def canonical_test_curve(curve_rows):
    has_val = any(row["split"] in {"val", "val_fast"} for row in curve_rows)
    if has_val:
        priority = {"val": 4, "val_fast": 3}
    else:
        priority = {"test": 3, "test_fast": 2}
    best = {}
    for row in curve_rows:
        if row["split"] not in priority:
            continue
        key = row["iteration"]
        old = best.get(key)
        if old is None or priority[row["split"]] >= priority.get(old["split"], 0):
            best[key] = row
    return [best[k] for k in sorted(best)]


def analyze_regression(curve_rows, threshold=DROP_THRESHOLD):
    test_rows = canonical_test_curve(curve_rows)
    if len(test_rows) < 2:
        return None

    peak = max(test_rows, key=lambda item: item["psnr"])
    final = test_rows[-1]
    first_bad = None
    for row in test_rows:
        if row["iteration"] > peak["iteration"] and peak["psnr"] - row["psnr"] >= threshold:
            first_bad = row
            break

    worst_from = None
    worst_to = None
    worst_drop = 0.0
    for prev, cur in zip(test_rows, test_rows[1:]):
        drop = prev["psnr"] - cur["psnr"]
        if drop > worst_drop:
            worst_drop = drop
            worst_from = prev
            worst_to = cur

    final_drop = peak["psnr"] - final["psnr"]
    flagged = final_drop >= threshold or worst_drop >= threshold
    return {
        "flagged": flagged,
        "peak_iter": peak["iteration"],
        "peak_psnr": peak["psnr"],
        "final_iter": final["iteration"],
        "final_psnr": final["psnr"],
        "final_drop": final_drop,
        "first_bad_iter": first_bad["iteration"] if first_bad else None,
        "first_bad_psnr": first_bad["psnr"] if first_bad else None,
        "worst_from_iter": worst_from["iteration"] if worst_from else None,
        "worst_to_iter": worst_to["iteration"] if worst_to else None,
        "worst_step_drop": worst_drop,
    }


def write_curve_svg(svg_path, title, curve_rows):
    if not curve_rows:
        return

    grouped = {}
    for row in curve_rows:
        if row["split"] not in {"val_fast", "val", "test_fast", "test", "train"}:
            continue
        grouped.setdefault(row["split"], []).append(row)
    if not grouped:
        return
    for rows in grouped.values():
        rows.sort(key=lambda item: item["iteration"])

    colors = {
        "val_fast": "#7c3aed",
        "val": "#9333ea",
        "test_fast": "#2563eb",
        "test": "#16a34a",
        "train": "#d97706",
    }
    width, height = 920, 420
    left, right, top, bottom = 66, 24, 36, 56
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_rows = [row for rows in grouped.values() for row in rows]
    min_x = min(row["iteration"] for row in all_rows)
    max_x = max(row["iteration"] for row in all_rows)
    min_y = min(row["psnr"] for row in all_rows)
    max_y = max(row["psnr"] for row in all_rows)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1.0
    pad = max(0.25, (max_y - min_y) * 0.08)
    min_y -= pad
    max_y += pad

    def x_pos(value):
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def y_pos(value):
        return top + (max_y - value) / (max_y - min_y) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="23" font-family="Arial" font-size="15" fill="#111827">{html.escape(title[:120])}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]

    for idx in range(5):
        y_val = min_y + (max_y - min_y) * idx / 4
        y = y_pos(y_val)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{y_val:.2f}</text>')
    for idx in range(5):
        x_val = int(round(min_x + (max_x - min_x) * idx / 4))
        x = x_pos(x_val)
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{x_val}</text>')

    for idx, split in enumerate(sorted(grouped)):
        color = colors.get(split, "#6b7280")
        y = top + 18 + idx * 18
        svg.append(f'<line x1="{left + 10}" y1="{y}" x2="{left + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{left + 34}" y="{y + 4}" font-family="Arial" font-size="12" fill="#374151">{split}</text>')

    for split, rows in grouped.items():
        color = colors.get(split, "#6b7280")
        points = " ".join(f'{x_pos(row["iteration"]):.2f},{y_pos(row["psnr"]):.2f}' for row in rows)
        if len(rows) > 1:
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for row in rows:
            svg.append(f'<circle cx="{x_pos(row["iteration"]):.2f}" cy="{y_pos(row["psnr"]):.2f}" r="2.5" fill="{color}"/>')

    test_rows = canonical_test_curve(curve_rows)
    if test_rows:
        peak = max(test_rows, key=lambda item: item["psnr"])
        final = test_rows[-1]
        svg.append(f'<text x="{left}" y="{height - 16}" font-family="Arial" font-size="12" fill="#111827">peak {peak["iteration"]}: {peak["psnr"]:.3f} | final {final["iteration"]}: {final["psnr"]:.3f} | drop {peak["psnr"] - final["psnr"]:.3f}</text>')
    svg.append("</svg>")
    Path(svg_path).write_text("\n".join(svg), encoding="utf-8")


def write_drop_chart(output_dir, regression_rows):
    rows = [row for row in regression_rows if row["analysis"] and row["analysis"]["flagged"]]
    rows.sort(key=lambda item: item["analysis"]["final_drop"], reverse=True)

    width = 980
    row_h = 26
    top = 48
    left = 330
    right = 36
    height = max(150, top + len(rows) * row_h + 42)
    max_drop = max([row["analysis"]["final_drop"] for row in rows] or [1.0])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="Arial" font-size="16" fill="#111827">Training Regression Drop</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        drop = row["analysis"]["final_drop"]
        bar_w = (width - left - right) * drop / max_drop
        label = f'{row["dataset"]}/{row["scene"]}'
        svg.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{html.escape(label[-64:])}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="17" fill="#dc2626" opacity="0.78"/>')
        svg.append(f'<text x="{left + bar_w + 6:.2f}" y="{y + 13}" font-family="Arial" font-size="11" fill="#111827">{drop:.3f}</text>')
    svg.append("</svg>")
    Path(output_dir, "training_regression_drop.svg").write_text("\n".join(svg), encoding="utf-8")


def format_metric(value, width=12):
    return f"{value:>{width}.6f}" if value is not None else f"{'N/A':>{width}}"


def summarize_dataset(dataset_path, dataset_name, drop_threshold):
    scenes = []
    regressions = []

    for scene_name in sorted(os.listdir(dataset_path)):
        scene_path = os.path.join(dataset_path, scene_name)
        if not os.path.isdir(scene_path):
            continue

        metric_file = os.path.join(scene_path, "metric.txt")
        if not os.path.exists(metric_file):
            print(f"  [跳过] 无 metric.txt: {scene_name}")
            continue

        metrics = parse_metric(metric_file)
        curve_file = os.path.join(scene_path, "eval_curve.txt")
        if not os.path.exists(curve_file):
            curve_file = os.path.join(scene_path, "eval_curve.csv")
        curve_rows = parse_eval_curve(curve_file)
        analysis = analyze_regression(curve_rows, drop_threshold)
        if curve_rows:
            write_curve_svg(os.path.join(scene_path, "eval_curve.svg"), f"{dataset_name}/{scene_name}", curve_rows)

        if metrics:
            scenes.append((scene_name, metrics, analysis))
            regressions.append({"dataset": dataset_name, "scene": scene_name, "analysis": analysis})
            flag = ""
            if analysis and analysis["flagged"]:
                flag = f"  [开倒车: peak {analysis['peak_iter']} -> final {analysis['final_iter']}, drop {analysis['final_drop']:.3f}]"
            print(f"  [OK] {scene_name}: {metrics}{flag}")

    if not scenes:
        print(f"  [警告] {dataset_name} 下没有找到任何有效指标，跳过。")
        return [], []

    averages = {}
    for key in METRIC_KEYS:
        vals = [m[key] for _, m, _ in scenes if key in m]
        averages[key] = sum(vals) / len(vals) if vals else None

    summary_path = os.path.join(dataset_path, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 96 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 96 + "\n")
        f.write(f"{'Scene':<20} {'PSNR':>12} {'SSIM':>12} {'LPIPS':>12} {'FPS':>10} {'PeakIter':>9} {'PeakPSNR':>10} {'Drop':>9} {'FirstBad':>9}\n")
        f.write("-" * 96 + "\n")
        for scene_name, metrics, analysis in scenes:
            peak_iter = analysis["peak_iter"] if analysis else ""
            peak_psnr = f"{analysis['peak_psnr']:.4f}" if analysis else ""
            drop = f"{analysis['final_drop']:.4f}" if analysis else ""
            first_bad = analysis["first_bad_iter"] if analysis and analysis["first_bad_iter"] is not None else ""
            f.write(
                f"{scene_name:<20} "
                f"{metrics.get('psnr', float('nan')):>12.6f} "
                f"{metrics.get('ssim', float('nan')):>12.6f} "
                f"{metrics.get('lpips', float('nan')):>12.6f} "
                f"{metrics.get('fps', float('nan')):>10.2f} "
                f"{str(peak_iter):>9} {str(peak_psnr):>10} {str(drop):>9} {str(first_bad):>9}\n"
            )
        f.write("-" * 96 + "\n")
        f.write(
            f"{'[Average]':<20} "
            f"{format_metric(averages.get('psnr'))} "
            f"{format_metric(averages.get('ssim'))} "
            f"{format_metric(averages.get('lpips'))} "
            f"{(averages.get('fps') or 0):>10.2f}\n"
        )
        f.write("=" * 96 + "\n")
        f.write(f"Total scenes: {len(scenes)}\n")

    print(f"  -> 已写入: {summary_path}\n")
    return scenes, regressions


def write_global_summary(output_dir, dataset_summaries, regression_rows, drop_threshold):
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 112 + "\n")
        f.write(f"Output: {output_dir}\n")
        f.write("=" * 112 + "\n\n")

        all_scenes = []
        for dataset_name, scenes in dataset_summaries:
            f.write(f"[Dataset] {dataset_name}\n")
            f.write(f"{'Scene':<20} {'PSNR':>12} {'SSIM':>12} {'LPIPS':>12} {'FPS':>10} {'PeakIter':>9} {'Drop':>9}\n")
            f.write("-" * 88 + "\n")
            for scene_name, metrics, analysis in scenes:
                all_scenes.append((dataset_name, scene_name, metrics, analysis))
                peak_iter = analysis["peak_iter"] if analysis else ""
                drop = f"{analysis['final_drop']:.4f}" if analysis else ""
                f.write(
                    f"{scene_name:<20} "
                    f"{metrics.get('psnr', float('nan')):>12.6f} "
                    f"{metrics.get('ssim', float('nan')):>12.6f} "
                    f"{metrics.get('lpips', float('nan')):>12.6f} "
                    f"{metrics.get('fps', float('nan')):>10.2f} "
                    f"{str(peak_iter):>9} {str(drop):>9}\n"
                )
            f.write("\n")

        f.write("[Overall Average]\n")
        for key in METRIC_KEYS:
            vals = [metrics[key] for _, _, metrics, _ in all_scenes if key in metrics]
            avg = sum(vals) / len(vals) if vals else None
            f.write(f"{key:<8}: {avg:.6f}\n" if avg is not None else f"{key:<8}: N/A\n")

    reg_path = os.path.join(output_dir, "training_regression_summary.txt")
    flagged = [row for row in regression_rows if row["analysis"] and row["analysis"]["flagged"]]
    flagged.sort(key=lambda item: item["analysis"]["final_drop"], reverse=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        f.write("=" * 118 + "\n")
        f.write(f"Training Regression Summary  (drop_threshold = {drop_threshold:.2f} PSNR)\n")
        f.write("=" * 118 + "\n")
        if not flagged:
            f.write("未发现超过阈值的开倒车场景。\n")
        else:
            f.write(f"{'Dataset':<18} {'Scene':<18} {'PeakIter':>9} {'PeakPSNR':>10} {'FinalIter':>9} {'FinalPSNR':>10} {'Drop':>9} {'FirstBad':>9} {'WorstStep':>17}\n")
            f.write("-" * 118 + "\n")
            for row in flagged:
                analysis = row["analysis"]
                worst = ""
                if analysis["worst_from_iter"] is not None:
                    worst = f"{analysis['worst_from_iter']}->{analysis['worst_to_iter']}(-{analysis['worst_step_drop']:.2f})"
                first_bad = analysis["first_bad_iter"] if analysis["first_bad_iter"] is not None else ""
                f.write(
                    f"{row['dataset']:<18} {row['scene']:<18} "
                    f"{analysis['peak_iter']:>9} {analysis['peak_psnr']:>10.4f} "
                    f"{analysis['final_iter']:>9} {analysis['final_psnr']:>10.4f} "
                    f"{analysis['final_drop']:>9.4f} {str(first_bad):>9} {worst:>17}\n"
                )
        f.write("=" * 118 + "\n")

    write_drop_chart(output_dir, regression_rows)
    print(f"-> 已写入总表: {summary_path}")
    print(f"-> 已写入开倒车表: {reg_path}")
    print(f"-> 已写入开倒车图: {os.path.join(output_dir, 'training_regression_drop.svg')}")


def main():
    parser = argparse.ArgumentParser(description="Collect metrics and training regression curves for PhysNorm-GS outputs.")
    parser.add_argument("output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--drop-threshold", type=float, default=DROP_THRESHOLD)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        print(f"[错误] 路径不存在: {output_dir}")
        return

    datasets = [
        d for d in sorted(os.listdir(output_dir))
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not datasets:
        print("[错误] output 目录下没有找到任何数据集文件夹。")
        return

    print(f"找到 {len(datasets)} 个数据集: {datasets}\n")
    dataset_summaries = []
    regression_rows = []
    for dataset_name in datasets:
        dataset_path = os.path.join(output_dir, dataset_name)
        print(f"[处理数据集] {dataset_name}")
        scenes, regressions = summarize_dataset(dataset_path, dataset_name, args.drop_threshold)
        if scenes:
            dataset_summaries.append((dataset_name, scenes))
            regression_rows.extend(regressions)

    write_global_summary(output_dir, dataset_summaries, regression_rows, args.drop_threshold)
    print("全部完成！")


if __name__ == "__main__":
    main()
