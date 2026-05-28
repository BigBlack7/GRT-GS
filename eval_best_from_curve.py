#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

METRIC_KEYS = ["psnr", "ssim", "lpips", "fps"]


def parse_curve(curve_path, select_splits):
    rows = []
    path = Path(curve_path)
    if not path.exists():
        return rows
    allowed = set(select_splits)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            split = parts[1]
            if split not in allowed:
                continue
            try:
                rows.append({"iteration": int(parts[0]), "split": split, "psnr": float(parts[3])})
            except ValueError:
                continue
    priority = {split: idx for idx, split in enumerate(select_splits, start=1)}
    by_iter = {}
    for row in rows:
        old = by_iter.get(row["iteration"])
        if old is None or priority[row["split"]] >= priority[old["split"]]:
            by_iter[row["iteration"]] = row
    return [by_iter[k] for k in sorted(by_iter)]


def parse_metric_text(text):
    metrics = {}
    for key in METRIC_KEYS:
        match = re.search(rf"{key}\s*:\s*([-+0-9.eE]+)", text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def read_metric(metric_path):
    path = Path(metric_path)
    if not path.exists():
        return {}
    return parse_metric_text(path.read_text(encoding="utf-8", errors="ignore"))


def copy_if_exists(src, dst):
    if Path(src).exists():
        shutil.copy2(src, dst)


def restore_file(backup_path, target_path):
    backup = Path(backup_path)
    target = Path(target_path)
    if backup.exists():
        shutil.copy2(backup, target)
        backup.unlink()
    elif target.exists():
        target.unlink()


def iter_exists(scene_path, iteration):
    return Path(scene_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply").exists()


def saved_iterations(scene_path):
    point_root = Path(scene_path, "point_cloud")
    if not point_root.exists():
        return set()
    values = set()
    for child in point_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if Path(child, "point_cloud.ply").exists():
            values.add(iteration)
    return values


def saved_curve_candidates(curve_rows, available_iters):
    return sorted(
        [row for row in curve_rows if row["iteration"] in available_iters],
        key=lambda item: item["psnr"],
        reverse=True,
    )


def select_best_saved_iteration(curve_rows, available_iters):
    if not available_iters:
        return None
    candidates = saved_curve_candidates(curve_rows, available_iters)
    if candidates:
        return candidates[0]
    peak = max(curve_rows, key=lambda item: item["psnr"])
    before_peak = sorted([it for it in available_iters if it <= peak["iteration"]], reverse=True)
    if before_peak:
        return {"iteration": before_peak[0], "split": "saved_fallback", "psnr": None}
    after_peak = sorted(available_iters)
    return {"iteration": after_peak[0], "split": "saved_fallback", "psnr": None}


def scene_iter_from_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("scene override must look like Dataset/scene=25000")
    scene_key, iter_text = value.split("=", 1)
    try:
        iteration = int(iter_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid iteration in {value}") from exc
    return scene_key.strip("/"), iteration


def collect_scenes(output_dir):
    output = Path(output_dir)
    scenes = []
    for dataset_dir in sorted(p for p in output.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            if Path(scene_dir, "cfg_args").exists():
                scenes.append((dataset_dir.name, scene_dir.name, scene_dir))
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Evaluate the best saved iteration selected from eval_curve.txt.")
    parser.add_argument("output_dir")
    parser.add_argument("--only-flagged", action="store_true", help="Only evaluate scenes whose curve drops by at least --drop-threshold.")
    parser.add_argument("--drop-threshold", type=float, default=1.0)
    parser.add_argument("--scene-iter", action="append", default=[], type=scene_iter_from_arg, help="Override iteration, e.g. GlossySynthetic/bell_weak=25000")
    parser.add_argument("--select-splits", default="test,test_fast", help="Comma-separated curve splits used for selecting the best iteration. Earlier names have lower priority.")
    parser.add_argument("--exact-peak-only", action="store_true", help="Skip scenes if the curve peak iteration was not saved. Default selects the best saved iteration.")
    parser.add_argument("--eval-top-k", type=int, default=1, help="Evaluate the top-K saved curve iterations offline and keep the best PSNR.")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--keep-metric", action="store_true", help="Keep metric.txt overwritten by the selected iteration. Default restores original metric.txt.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    eval_py = script_dir / "eval.py"
    overrides = dict(args.scene_iter)
    select_splits = [item.strip() for item in args.select_splits.split(",") if item.strip()]
    rows = []

    for dataset, scene, scene_path in collect_scenes(output_dir):
        curve_rows = parse_curve(scene_path / "eval_curve.txt", select_splits)
        if not curve_rows:
            continue
        peak = max(curve_rows, key=lambda item: item["psnr"])
        final = curve_rows[-1]
        drop = peak["psnr"] - final["psnr"]
        scene_key = f"{dataset}/{scene}"
        available_iters = saved_iterations(scene_path)
        if scene_key in overrides:
            selected_candidates = [{"iteration": overrides[scene_key], "split": "override", "psnr": None}]
        elif args.exact_peak_only:
            selected_candidates = [peak]
        else:
            selected = select_best_saved_iteration(curve_rows, available_iters)
            if selected is None:
                print(f"[跳过] {scene_key}: 没有找到可评估保存点")
                continue
            saved_candidates = saved_curve_candidates(curve_rows, available_iters)
            if saved_candidates:
                selected_candidates = saved_candidates[: max(1, args.eval_top_k)]
            else:
                selected_candidates = [selected]
        if args.only_flagged and scene_key not in overrides and drop < args.drop_threshold:
            continue

        metric_path = scene_path / "metric.txt"
        result_path = scene_path / "results.json"
        metric_backup = scene_path / ".metric.before_best_eval.txt"
        result_backup = scene_path / ".results.before_best_eval.json"
        copy_if_exists(metric_path, metric_backup)
        copy_if_exists(result_path, result_backup)

        candidate_results = []
        for selected in selected_candidates:
            selected_iter = selected["iteration"]
            if not iter_exists(scene_path, selected_iter):
                print(f"[跳过] {scene_key}: 找不到保存点 iteration_{selected_iter}")
                continue
            cmd = [sys.executable, str(eval_py), "-m", str(scene_path), "--iteration", str(selected_iter)]
            if dataset != "RefReal":
                cmd.append("--white_background")
            if args.save_images:
                cmd.append("--save_images")
            print(f"[评估] {scene_key} @ {selected_iter}")
            subprocess.run(cmd, check=True)

            metrics = read_metric(metric_path)
            best_metric_path = scene_path / f"metric_best_{selected_iter}.txt"
            best_result_path = scene_path / f"results_best_{selected_iter}.json"
            copy_if_exists(metric_path, best_metric_path)
            copy_if_exists(result_path, best_result_path)
            candidate_results.append(
                {
                    "iteration": selected_iter,
                    "split": selected.get("split"),
                    "curve_psnr": selected.get("psnr"),
                    **metrics,
                }
            )

        if not args.keep_metric:
            restore_file(metric_backup, metric_path)
            restore_file(result_backup, result_path)

        if not candidate_results:
            continue
        best_candidate = max(candidate_results, key=lambda item: item.get("psnr", float("-inf")))

        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "selected_iter": best_candidate["iteration"],
                "selected_split": best_candidate.get("split"),
                "selected_curve_psnr": best_candidate.get("curve_psnr"),
                "curve_peak_iter": peak["iteration"],
                "curve_peak_psnr": peak["psnr"],
                "curve_final_iter": final["iteration"],
                "curve_final_psnr": final["psnr"],
                "curve_drop": drop,
                "offline_candidates": candidate_results,
                **{key: best_candidate[key] for key in METRIC_KEYS if key in best_candidate},
            }
        )

    summary_path = output_dir / "best_iteration_eval_summary.txt"
    json_path = output_dir / "best_iteration_eval_summary.json"
    averages = {}
    for key in METRIC_KEYS:
        vals = [row[key] for row in rows if key in row]
        averages[key] = sum(vals) / len(vals) if vals else None
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"Best-Iteration Offline Evaluation: {output_dir}\n")
        f.write(f"Selection splits: {', '.join(select_splits)}\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Dataset':<18} {'Scene':<24} {'Iter':>8} {'SelCurve':>10} {'CurvePeak':>10} {'CurveFinal':>11} {'Drop':>8} {'PSNR':>12} {'SSIM':>12} {'LPIPS':>12}\n")
        f.write("-" * 120 + "\n")
        for row in rows:
            selected_curve = row.get("selected_curve_psnr")
            selected_curve_text = f"{selected_curve:.4f}" if selected_curve is not None else "N/A"
            f.write(
                f"{row['dataset']:<18} {row['scene']:<24} "
                f"{row['selected_iter']:>8} {selected_curve_text:>10} {row['curve_peak_psnr']:>10.4f} {row['curve_final_psnr']:>11.4f} {row['curve_drop']:>8.4f} "
                f"{row.get('psnr', float('nan')):>12.6f} {row.get('ssim', float('nan')):>12.6f} {row.get('lpips', float('nan')):>12.6f}\n"
            )
        if rows:
            f.write("-" * 120 + "\n")
            f.write(
                f"{'[Average]':<18} {'':<24} {'':>8} {'':>10} {'':>10} {'':>11} {'':>8} "
                f"{(averages.get('psnr') if averages.get('psnr') is not None else float('nan')):>12.6f} "
                f"{(averages.get('ssim') if averages.get('ssim') is not None else float('nan')):>12.6f} "
                f"{(averages.get('lpips') if averages.get('lpips') is not None else float('nan')):>12.6f}\n"
            )
        f.write("=" * 120 + "\n")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"output_dir": str(output_dir), "select_splits": select_splits, "averages": averages, "scenes": rows}, f, indent=2, ensure_ascii=False)
    print(f"-> 已写入: {summary_path}")
    print(f"-> 已写入: {json_path}")


if __name__ == "__main__":
    main()
