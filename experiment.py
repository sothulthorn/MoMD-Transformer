"""
Systematic hyperparameter search for MoMD Transformer.

Phase 1: One-variable-at-a-time sweeps (1 repeat each, ~23 runs)
Phase 2: Targeted combinations based on Phase 1 results (2 repeats each)
Phase 3: Top-N configs from Phase 1 with full 10 repeats for mean±std comparison

Usage:
  python experiment.py --dataset pu --phase 1 --repeats 1
  python experiment.py --dataset pu --phase 2 --repeats 2
  python experiment.py --dataset pmsm --phase 3 --repeats 10 --top_n 5
"""

import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np

import config
from train import run_experiment


# ==============================================================================
# Phase 1: One-variable-at-a-time sweeps
# ==============================================================================
# Baseline values (defaults from config)
BASELINE = {
    "lambda_gkt": 1.0,
    "lambda_msm": 1.0,
    "lr": 1e-4,
    "epochs": 200,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "mask_ratio": 0.15,
}

PHASE1_SWEEPS = [
    ("lambda_gkt", [0.01, 0.1, 0.5, 1.0, 2.0]),
    ("lr", [1e-3, 5e-4, 1e-4, 5e-5]),
    ("epochs", [100, 200, 300]),
    ("lambda_msm", [0.0, 0.1, 0.5, 1.0]),
    ("batch_size", [32, 64, 128]),
    ("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
]


def apply_config(params):
    """Monkey-patch config module with experiment parameters."""
    config.LAMBDA_GKT = params["lambda_gkt"]
    config.LAMBDA_MSM = params["lambda_msm"]
    config.MASK_RATIO = params["mask_ratio"]
    config.WEIGHT_DECAY = params["weight_decay"]


def restore_config():
    """Restore config to baseline defaults."""
    apply_config(BASELINE)


def make_args(params, dataset, num_workers):
    """Create an args namespace compatible with run_experiment()."""
    return SimpleNamespace(
        dataset=dataset,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        num_workers=num_workers,
    )


def run_single(params, dataset, seed, run_dir, label_names, num_workers):
    """Run a single experiment with given params, return results dict."""
    apply_config(params)
    args = make_args(params, dataset, num_workers)
    results, _ = run_experiment(args, seed, run_dir, label_names)
    restore_config()
    return results


def generate_phase1_configs():
    """Yield (description, params_dict) for each Phase 1 run."""
    for sweep_var, values in PHASE1_SWEEPS:
        for val in values:
            params = dict(BASELINE)
            params[sweep_var] = val
            desc = f"{sweep_var}={val}"
            yield desc, params


def run_phase1(dataset, repeats, output_dir, num_workers):
    """Run Phase 1 one-variable-at-a-time sweeps."""
    data_config = config.PU_CONFIG if dataset == "pu" else config.PMSM_CONFIG
    label_names = data_config["label_names"]

    csv_path = os.path.join(output_dir, f"phase1_{dataset}.csv")
    configs = list(generate_phase1_configs())
    total_runs = len(configs) * repeats

    print(f"Phase 1: {len(configs)} configs x {repeats} repeats = {total_runs} runs")
    print(f"Results will be saved to {csv_path}\n")

    rows = []
    run_idx = 0

    for desc, params in configs:
        for rep in range(1, repeats + 1):
            run_idx += 1
            seed = config.SEED + rep - 1
            run_dir = os.path.join(output_dir, f"phase1_{dataset}", desc, f"rep_{rep}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {desc} (rep {rep}, seed {seed})")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                results = run_single(params, dataset, seed, run_dir, label_names,
                                     num_workers)
                elapsed = time.time() - t0

                row = {
                    "config": desc,
                    "rep": rep,
                    "seed": seed,
                    "current_acc": results["current"],
                    "vibration_acc": results["vibration"],
                    "both_acc": results["both"],
                    "time_s": f"{elapsed:.1f}",
                    **{k: v for k, v in params.items()},
                }
                rows.append(row)
                print(f"  -> current={results['current']:.2f}%, "
                      f"vib={results['vibration']:.2f}%, "
                      f"both={results['both']:.2f}% ({elapsed:.0f}s)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  -> FAILED: {e} ({elapsed:.0f}s)")
                row = {
                    "config": desc,
                    "rep": rep,
                    "seed": seed,
                    "current_acc": "FAIL",
                    "vibration_acc": "FAIL",
                    "both_acc": "FAIL",
                    "time_s": f"{elapsed:.1f}",
                    **{k: v for k, v in params.items()},
                }
                rows.append(row)

            # Write CSV after each run (incremental save)
            _write_csv(csv_path, rows)

    restore_config()
    print(f"\nPhase 1 complete. Results saved to {csv_path}")
    return rows


def find_best_phase1(csv_path):
    """Parse phase1 CSV and return best params per sweep variable."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    best_per_var = {}
    for row in rows:
        if row["current_acc"] == "FAIL":
            continue
        desc = row["config"]
        var_name = desc.split("=")[0]
        cur_acc = float(row["current_acc"])

        if var_name not in best_per_var or cur_acc > best_per_var[var_name]["current_acc"]:
            best_per_var[var_name] = {
                "current_acc": cur_acc,
                "config": desc,
                "params": {k: row[k] for k in BASELINE},
            }

    return best_per_var


def generate_phase2_configs(phase1_csv):
    """Generate Phase 2 configs based on Phase 1 results."""
    best = find_best_phase1(phase1_csv)

    print("Phase 1 best per variable:")
    for var, info in best.items():
        print(f"  {var}: {info['config']} -> current={info['current_acc']:.2f}%")

    # Config 1: Combined best from Phase 1
    combined = dict(BASELINE)
    for var, info in best.items():
        raw = info["params"][var]
        combined[var] = type(BASELINE[var])(raw)
    yield "combined_best", combined

    # Config 2: Low GKT + higher LR
    low_gkt_high_lr = dict(combined)
    low_gkt_high_lr["lambda_gkt"] = min(0.1, combined.get("lambda_gkt", 0.1))
    if "lr" in best:
        best_lr = type(BASELINE["lr"])(best["lr"]["params"]["lr"])
        low_gkt_high_lr["lr"] = max(best_lr, 5e-4)
    else:
        low_gkt_high_lr["lr"] = 5e-4
    yield "low_gkt_high_lr", low_gkt_high_lr

    # Config 3: No GKT ablation
    no_gkt = dict(combined)
    no_gkt["lambda_gkt"] = 0.0
    yield "no_gkt", no_gkt

    # Config 4: No GKT + no MSM (classification only)
    cls_only = dict(combined)
    cls_only["lambda_gkt"] = 0.0
    cls_only["lambda_msm"] = 0.0
    yield "cls_only", cls_only


def run_phase2(dataset, repeats, output_dir, num_workers):
    """Run Phase 2 targeted combinations."""
    data_config = config.PU_CONFIG if dataset == "pu" else config.PMSM_CONFIG
    label_names = data_config["label_names"]

    phase1_csv = os.path.join(output_dir, f"phase1_{dataset}.csv")
    if not os.path.exists(phase1_csv):
        print(f"ERROR: Phase 1 results not found at {phase1_csv}")
        print("Run Phase 1 first: python experiment.py --dataset {dataset} --phase 1")
        return []

    csv_path = os.path.join(output_dir, f"phase2_{dataset}.csv")
    configs = list(generate_phase2_configs(phase1_csv))
    total_runs = len(configs) * repeats

    print(f"\nPhase 2: {len(configs)} configs x {repeats} repeats = {total_runs} runs")
    print(f"Results will be saved to {csv_path}\n")

    rows = []
    run_idx = 0

    for desc, params in configs:
        for rep in range(1, repeats + 1):
            run_idx += 1
            seed = config.SEED + rep - 1
            run_dir = os.path.join(output_dir, f"phase2_{dataset}", desc, f"rep_{rep}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {desc} (rep {rep}, seed {seed})")
            print(f"  params: {params}")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                results = run_single(params, dataset, seed, run_dir, label_names,
                                     num_workers)
                elapsed = time.time() - t0

                row = {
                    "config": desc,
                    "rep": rep,
                    "seed": seed,
                    "current_acc": results["current"],
                    "vibration_acc": results["vibration"],
                    "both_acc": results["both"],
                    "time_s": f"{elapsed:.1f}",
                    **{k: v for k, v in params.items()},
                }
                rows.append(row)
                print(f"  -> current={results['current']:.2f}%, "
                      f"vib={results['vibration']:.2f}%, "
                      f"both={results['both']:.2f}% ({elapsed:.0f}s)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  -> FAILED: {e} ({elapsed:.0f}s)")
                row = {
                    "config": desc,
                    "rep": rep,
                    "seed": seed,
                    "current_acc": "FAIL",
                    "vibration_acc": "FAIL",
                    "both_acc": "FAIL",
                    "time_s": f"{elapsed:.1f}",
                    **{k: v for k, v in params.items()},
                }
                rows.append(row)

            _write_csv(csv_path, rows)

    restore_config()
    print(f"\nPhase 2 complete. Results saved to {csv_path}")
    return rows


def rank_phase1_configs(csv_path, top_n=5):
    """Rank all Phase 1 configs by current_acc (descending), return top N."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Average across repeats per config (handles repeats > 1)
    config_accs = {}
    config_params = {}
    for row in rows:
        if row["current_acc"] == "FAIL":
            continue
        desc = row["config"]
        cur = float(row["current_acc"])
        vib = float(row["vibration_acc"])
        both = float(row["both_acc"])
        config_accs.setdefault(desc, []).append((cur, vib, both))
        config_params[desc] = {k: row[k] for k in BASELINE}

    ranked = []
    for desc, accs in config_accs.items():
        cur_mean = np.mean([a[0] for a in accs])
        vib_mean = np.mean([a[1] for a in accs])
        both_mean = np.mean([a[2] for a in accs])
        ranked.append((desc, cur_mean, vib_mean, both_mean, config_params[desc]))

    # Sort by current_acc descending, then both_acc descending as tiebreak
    ranked.sort(key=lambda x: (x[1], x[3]), reverse=True)
    return ranked[:top_n]


def run_phase3(dataset, repeats, output_dir, num_workers, top_n):
    """Run Phase 3: top-N Phase 1 configs with full repeats for mean±std."""
    data_config = config.PU_CONFIG if dataset == "pu" else config.PMSM_CONFIG
    label_names = data_config["label_names"]

    phase1_csv = os.path.join(output_dir, f"phase1_{dataset}.csv")
    if not os.path.exists(phase1_csv):
        print(f"ERROR: Phase 1 results not found at {phase1_csv}")
        print(f"Run Phase 1 first: python experiment.py --dataset {dataset} --phase 1")
        return []

    top_configs = rank_phase1_configs(phase1_csv, top_n)

    print(f"Phase 3: Top {len(top_configs)} configs from Phase 1 (by current_acc):")
    for desc, cur, vib, both, _ in top_configs:
        print(f"  {desc:30s}  cur={cur:.2f}%  vib={vib:.2f}%  both={both:.2f}%")

    csv_path = os.path.join(output_dir, f"phase3_{dataset}.csv")
    summary_path = os.path.join(output_dir, f"phase3_{dataset}_summary.csv")
    total_runs = len(top_configs) * repeats

    print(f"\n{len(top_configs)} configs x {repeats} repeats = {total_runs} runs")
    print(f"Results: {csv_path}")
    print(f"Summary: {summary_path}\n")

    rows = []
    summary_rows = []
    run_idx = 0

    for desc, _, _, _, raw_params in top_configs:
        params = {k: type(BASELINE[k])(raw_params[k]) for k in BASELINE}
        config_results = {"current": [], "vibration": [], "both": []}

        for rep in range(1, repeats + 1):
            run_idx += 1
            seed = config.SEED + rep - 1
            run_dir = os.path.join(output_dir, f"phase3_{dataset}", desc, f"rep_{rep}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] {desc} (rep {rep}/{repeats}, seed {seed})")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                results = run_single(params, dataset, seed, run_dir, label_names,
                                     num_workers)
                elapsed = time.time() - t0

                for mode in ["current", "vibration", "both"]:
                    config_results[mode].append(results[mode])

                row = {
                    "config": desc,
                    "rep": rep,
                    "seed": seed,
                    "current_acc": results["current"],
                    "vibration_acc": results["vibration"],
                    "both_acc": results["both"],
                    "time_s": f"{elapsed:.1f}",
                    **{k: v for k, v in params.items()},
                }
                rows.append(row)
                print(f"  -> current={results['current']:.2f}%, "
                      f"vib={results['vibration']:.2f}%, "
                      f"both={results['both']:.2f}% ({elapsed:.0f}s)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  -> FAILED: {e} ({elapsed:.0f}s)")
                row = {
                    "config": desc,
                    "rep": rep,
                    "seed": seed,
                    "current_acc": "FAIL",
                    "vibration_acc": "FAIL",
                    "both_acc": "FAIL",
                    "time_s": f"{elapsed:.1f}",
                    **{k: v for k, v in params.items()},
                }
                rows.append(row)

            _write_csv(csv_path, rows)

        # Compute summary for this config
        if config_results["current"]:
            cur_arr = np.array(config_results["current"])
            vib_arr = np.array(config_results["vibration"])
            both_arr = np.array(config_results["both"])
            summary_row = {
                "config": desc,
                "n_runs": len(cur_arr),
                "current_mean": f"{cur_arr.mean():.2f}",
                "current_std": f"{cur_arr.std():.2f}",
                "vibration_mean": f"{vib_arr.mean():.2f}",
                "vibration_std": f"{vib_arr.std():.2f}",
                "both_mean": f"{both_arr.mean():.2f}",
                "both_std": f"{both_arr.std():.2f}",
                **{k: v for k, v in params.items()},
            }
            summary_rows.append(summary_row)
            _write_csv(summary_path, summary_rows)

            print(f"\n  >> {desc}: "
                  f"current={cur_arr.mean():.2f}±{cur_arr.std():.2f}  "
                  f"vib={vib_arr.mean():.2f}±{vib_arr.std():.2f}  "
                  f"both={both_arr.mean():.2f}±{both_arr.std():.2f}")

    restore_config()
    print(f"\nPhase 3 complete.")
    print(f"  Per-run results: {csv_path}")
    print(f"  Summary (mean±std): {summary_path}")

    # Print final comparison table
    if summary_rows:
        print(f"\n{'='*80}")
        print(f"{'Config':30s} {'Current':>16s} {'Vibration':>16s} {'Both':>16s}")
        print(f"{'='*80}")
        for s in summary_rows:
            print(f"{s['config']:30s} "
                  f"{s['current_mean']:>6s}±{s['current_std']:<6s}  "
                  f"{s['vibration_mean']:>6s}±{s['vibration_std']:<6s}  "
                  f"{s['both_mean']:>6s}±{s['both_std']:<6s}")
        print(f"{'='*80}")

    return rows


def _write_csv(path, rows):
    """Write rows to CSV, creating header from first row's keys."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="MoMD Transformer Hyperparameter Search")
    parser.add_argument("--dataset", type=str, default="pu", choices=["pu", "pmsm"])
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Phase 1: one-var sweeps, Phase 2: targeted combos, "
                             "Phase 3: top-N with full repeats")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per config (default: 1)")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Phase 3: number of top configs to evaluate (default: 5)")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--output_dir", type=str, default="results/experiments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Phase: {args.phase}")
    print(f"Repeats: {args.repeats}")
    print(f"Output: {args.output_dir}")

    if args.phase == 1:
        run_phase1(args.dataset, args.repeats, args.output_dir, args.num_workers)
    elif args.phase == 2:
        run_phase2(args.dataset, args.repeats, args.output_dir, args.num_workers)
    else:
        run_phase3(args.dataset, args.repeats, args.output_dir, args.num_workers,
                   args.top_n)


if __name__ == "__main__":
    main()
