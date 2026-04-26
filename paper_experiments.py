#!/usr/bin/env python3
"""
Paper Experiments Runner for HyBMSearch

Replicates the experiments described in paper.tex for uniform (sequential) datasets
at sizes: 100M, 1B, 5B, 10B. For each size, it:
  1) Generates data (sequential array, random targets)
  2) Runs GA+Bayesian optimization to find best parameters
  3) Benchmarks HyBMSearch vs baselines (NumPy searchsorted, Parallel Binary,
     Parallel Interpolation, Parallel Fibonacci)
  4) Validates correctness vs NumPy
  5) Saves plot-ready data and JSON artifacts in a dedicated results directory

Dependencies: numpy, hybmsearch (installed/in workspace), and the same libraries
as used in comprehensive_benchmark.py. Intended for HPC execution.
"""

import os
import sys
import json
import time
import math
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

# Ensure local package import
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hybmsearch import SearchConfig, optimize_search_parameters
from hybmsearch.benchmarking import (
    benchmark_search,
    benchmark_numpy_search,
    benchmark_parallel_binary_search,
    benchmark_parallel_interpolation_search,
    benchmark_parallel_fibonacci_search,
    validate_results,
)
from hybmsearch.utils import create_test_data, setup_logger


class _GAHistoryLogHandler(logging.Handler):
    """Capture GA per-generation stats from optimization logs.

    Supports two patterns:
    1) Summary lines: "Generation X: Min Time = ..., Avg Time = ..., ..."
    2) Evaluation lines: "Evaluated (Gen=X): {...}, Time=...s" with per-evaluation times
       which we aggregate into per-generation stats later.
    """

    def __init__(self):
        super().__init__()
        self.records: List[Dict[str, Any]] = []
        self.times_by_gen: Dict[int, List[float]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        # Pattern 1: Summary line
        if msg.startswith("Generation ") and "Min Time" in msg and "Avg Time" in msg:
            try:
                parts = msg.split(':', 1)
                gen_num = int(parts[0].replace('Generation', '').strip())
                rest = parts[1]
                kv = {}
                for token in rest.split(','):
                    if '=' in token:
                        k, v = token.split('=')
                        k = k.strip()
                        v = v.strip().rstrip('s')
                        kv[k] = float(v)
                self.records.append({
                    'generation': gen_num,
                    'min_time': kv.get('Min Time'),
                    'avg_time': kv.get('Avg Time'),
                    'max_time': kv.get('Max Time'),
                    'std_time': kv.get('Std Dev'),
                })
            except Exception:
                pass
            return

        # Pattern 2: Per-evaluation line (e.g., "Evaluated (Gen=0): ..., Time=15.939960s")
        if "Evaluated (Gen=" in msg and ", Time=" in msg:
            try:
                # Extract generation
                seg = msg.split("Evaluated (Gen=", 1)[1]
                gen_str = seg.split(")", 1)[0]
                gen_num = int(gen_str)
                # Extract time seconds
                tseg = msg.split(", Time=", 1)[1]
                t_str = tseg.split('s', 1)[0]
                t_val = float(t_str)
                self.times_by_gen.setdefault(gen_num, []).append(t_val)
            except Exception:
                pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_experiment_for_size(size: int, target_count: int, 
                            pop_size: int, ngen: int, base_outdir: str,
                            num_workers: int = None) -> Dict[str, Any]:
    """Run the full pipeline for a given array size."""
    label = f"{size:_}".replace('_', '')
    outdir = os.path.join(base_outdir, f"size_{label}")
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, 'params'))
    ensure_dir(os.path.join(outdir, 'plots_data'))
    ensure_dir(os.path.join(outdir, 'benchmarks'))

    # 1) Data generation (sequential)
    arr, targets = create_test_data(size=size, target_count=target_count, data_type="sequential")
    # Use int32 when safe, otherwise int64 to prevent overflow at multi‑billion sizes
    safe_dtype = np.int32 if size <= np.iinfo(np.int32).max else np.int64
    arr = np.asarray(arr, dtype=safe_dtype)
    targets = np.asarray(targets, dtype=safe_dtype)

    # Persist minimal metadata for reproducibility
    meta = {
        'size': int(size),
        'target_count': int(target_count),
        'distribution': 'sequential',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    with open(os.path.join(outdir, 'plots_data', 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 2) Optimization
    ga_logger = logging.getLogger('hybmsearch')
    ga_handler = _GAHistoryLogHandler()
    ga_logger.addHandler(ga_handler)
    try:
        best_params, eval_cache = optimize_search_parameters(
            arr, targets, pop_size=pop_size, ngen=ngen
        )
    finally:
        ga_logger.removeHandler(ga_handler)

    # Save optimized parameters and raw evaluation cache
    with open(os.path.join(outdir, 'params', 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    # eval_cache keys are tuples of sorted (k,v); convert to list for JSON
    def _to_py(v):
        try:
            import numpy as _np
            if isinstance(v, _np.generic):
                return v.item()
        except Exception:
            pass
        if isinstance(v, (list, tuple)):
            return [_to_py(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _to_py(val) for k, val in v.items()}
        return v

    cache_list = []
    for k_tuple, fitness in eval_cache.items():
        params_dict = {str(k): _to_py(v) for k, v in dict(k_tuple).items()}
        entry = params_dict.copy()
        entry['time'] = float(fitness[0])
        cache_list.append(entry)
    with open(os.path.join(outdir, 'params', 'eval_cache.json'), 'w') as f:
        json.dump(cache_list, f, indent=2)

    # Save GA history (per-generation) for plotting.
    # Prefer explicit summary records if present; otherwise aggregate per-evaluation logs.
    ga_hist_path = os.path.join(outdir, 'plots_data', 'ga_history.csv')
    with open(ga_hist_path, 'w') as f:
        f.write('generation,min_time,avg_time,max_time,std_time\n')
        if ga_handler.records:
            for rec in sorted(ga_handler.records, key=lambda r: r['generation']):
                f.write(f"{rec['generation']},{rec['min_time']},{rec['avg_time']},{rec['max_time']},{rec['std_time']}\n")
        elif ga_handler.times_by_gen:
            for gen in sorted(ga_handler.times_by_gen.keys()):
                times = np.array(ga_handler.times_by_gen[gen], dtype=float)
                if times.size == 0:
                    continue
                f.write(f"{gen},{times.min()},{times.mean()},{times.max()},{times.std(ddof=0)}\n")

    # 3) Benchmark with optimized params and baselines
    cfg = SearchConfig(**best_params)

    t_hybm, res_hybm = benchmark_search(arr, targets, config=cfg)
    t_np, res_np = benchmark_numpy_search(arr, targets)

    t_bin, r_bin = benchmark_parallel_binary_search(arr, targets, num_workers=num_workers)
    t_int, r_int = benchmark_parallel_interpolation_search(arr, targets, num_workers=num_workers)
    t_fib, r_fib = benchmark_parallel_fibonacci_search(arr, targets, num_workers=num_workers)

    # Validation vs NumPy
    mismatches_hybm = validate_results(res_hybm, res_np, targets, arr)
    mismatches_bin = validate_results(r_bin, res_np, targets, arr)
    mismatches_int = validate_results(r_int, res_np, targets, arr)
    mismatches_fib = validate_results(r_fib, res_np, targets, arr)

    # Save benchmark summary (plot-ready)
    times = {
        'HyBMSearch': float(t_hybm),
        'NumPy_searchsorted': float(t_np),
        'Parallel_Binary_Py': float(t_bin),
        'Parallel_Interpolation_Py': float(t_int),
        'Parallel_Fibonacci_Py': float(t_fib),
    }
    with open(os.path.join(outdir, 'plots_data', 'bar_times.csv'), 'w') as f:
        f.write('method,time_seconds\n')
        for k, v in times.items():
            f.write(f"{k},{v}\n")

    # Save detailed results
    results = {
        'times': times,
        'mismatches': {
            'HyBMSearch': int(mismatches_hybm),
            'Parallel_Binary_Py': int(mismatches_bin),
            'Parallel_Interpolation_Py': int(mismatches_int),
            'Parallel_Fibonacci_Py': int(mismatches_fib),
        },
        'found_counts': {
            'HyBMSearch': int(np.sum(res_hybm != -1)),
            'NumPy_searchsorted': int(np.sum(res_np != -1)),
            'Parallel_Binary_Py': int(np.sum(r_bin != -1)),
            'Parallel_Interpolation_Py': int(np.sum(r_int != -1)),
            'Parallel_Fibonacci_Py': int(np.sum(r_fib != -1)),
        },
        'best_params': best_params,
    }
    with open(os.path.join(outdir, 'benchmarks', 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return {
        'outdir': outdir,
        'results': results,
    }


def main():
    # Configure a top-level logger file
    setup_logger("paper_experiments.log")

    # Experiment sizes (reduced): test only on 5B and 10B
    sizes = [
        5_000_000_000,
        10_000_000_000,
    ]
    # Targets as in paper: q = max(100_000, n/10)
    def targets_for(n: int) -> int:
        return int(max(100_000, n // 10))

    # GA settings (tune as desired on HPC)
    POP_SIZE = 50
    NGEN = 10

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    base_outdir = os.path.join(PROJECT_ROOT, f"paper_results_{timestamp}")
    ensure_dir(base_outdir)

    summary = {}
    for n in sizes:
        q = targets_for(n)
        info = run_experiment_for_size(n, q, POP_SIZE, NGEN, base_outdir)
        summary[str(n)] = info['results']

    # Global summary file
    with open(os.path.join(base_outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"All experiments completed. Results saved under: {base_outdir}")


if __name__ == '__main__':
    main()


