#!/usr/bin/env python3
"""
1 Billion Element Benchmark with int32 Memory Efficiency

Tests HyBMSearch performance on 1 billion elements using int32 for memory efficiency.
Tests 5 different data distributions to demonstrate algorithm robustness.
"""

import os
import sys
import json
import time
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
from hybmsearch.utils import setup_logger


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_data_distribution(size: int, target_count: int, distribution_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data for different distributions using int32 for memory efficiency.
    
    Args:
        size: Array size (1 billion)
        target_count: Number of targets
        distribution_type: Type of distribution
        
    Returns:
        Tuple of (array, targets) both as int32
    """
    # Use int32 for memory efficiency (1B elements = 4GB memory)
    dtype = np.int32
    int32_max = np.iinfo(dtype).max  # 2,147,483,647
    
    if distribution_type == "sequential":
        # Perfect sequential data: 0, 1, 2, ..., size-1
        arr = np.arange(size, dtype=dtype)
        # Generate targets within range
        targets_found = np.random.choice(arr, size=target_count // 2, replace=True)
        targets_random = np.random.randint(0, size, size=target_count - len(targets_found), dtype=dtype)
        targets = np.concatenate([targets_found, targets_random])
        np.random.shuffle(targets)
        
    elif distribution_type == "random_uniform":
        # Random uniform distribution
        max_val = min(size * 2, int32_max)
        arr = np.sort(np.random.randint(0, max_val, size, dtype=dtype))
        targets = np.random.randint(arr[0], min(arr[-1] + 1, int32_max), target_count, dtype=dtype)
        
    elif distribution_type == "sparse":
        # Sparse data with large gaps
        max_val = min(size * 10, int32_max)
        arr = np.sort(np.random.randint(0, max_val, size, dtype=dtype))
        targets = np.random.randint(arr[0], min(arr[-1] + 1, int32_max), target_count, dtype=dtype)
        
    elif distribution_type == "exponential":
        # Exponential distribution (scaled to fit int32)
        scale = min(1000, int32_max // max(1, size))
        arr = np.sort(np.random.exponential(scale=scale, size=size).astype(dtype))
        # Ensure values fit in int32
        arr = np.clip(arr, 0, int32_max)
        targets = np.random.randint(arr[0], min(arr[-1] + 1, int32_max), target_count, dtype=dtype)
        
    elif distribution_type == "bimodal":
        # Bimodal distribution (two peaks)
        peak1 = np.random.normal(size * 0.2, size * 0.05, size // 2)
        peak2 = np.random.normal(size * 0.8, size * 0.05, size // 2)
        arr = np.sort(np.concatenate([peak1, peak2]).astype(dtype))
        # Ensure values fit in int32
        arr = np.clip(arr, 0, int32_max)
        targets = np.random.randint(arr[0], min(arr[-1] + 1, int32_max), target_count, dtype=dtype)
        
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return arr, targets


def run_benchmark_for_distribution(distribution_type: str, size: int, target_count: int, 
                                 pop_size: int, ngen: int, base_outdir: str) -> Dict[str, Any]:
    """Run benchmark for a specific data distribution."""
    
    outdir = os.path.join(base_outdir, f"distribution_{distribution_type}")
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, 'params'))
    ensure_dir(os.path.join(outdir, 'benchmarks'))
    
    print(f"\n=== Benchmarking {distribution_type.upper()} Distribution ===")
    print(f"Array size: {size:,}, Target count: {target_count:,}")
    
    # 1) Generate data
    print("Generating data...")
    arr, targets = generate_data_distribution(size, target_count, distribution_type)
    print(f"Array range: [{arr[0]:,}, {arr[-1]:,}]")
    print(f"Memory usage: {arr.nbytes / (1024**3):.2f} GB")
    
    # Save metadata
    meta = {
        'distribution': distribution_type,
        'size': int(size),
        'target_count': int(target_count),
        'dtype': 'int32',
        'array_range': [int(arr[0]), int(arr[-1])],
        'memory_gb': float(arr.nbytes / (1024**3)),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # 2) Optimization
    print("Running GA optimization...")
    start_opt = time.time()
    best_params, eval_cache = optimize_search_parameters(
        arr, targets, pop_size=pop_size, ngen=ngen
    )
    opt_time = time.time() - start_opt
    print(f"Optimization completed in {opt_time:.2f}s")
    
    # Save optimized parameters
    with open(os.path.join(outdir, 'params', 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 3) Benchmark all methods
    print("Running benchmarks...")
    cfg = SearchConfig(**best_params)
    
    # HyBMSearch with optimized params
    t_hybm, res_hybm = benchmark_search(arr, targets, config=cfg)
    
    # Baseline methods
    t_np, res_np = benchmark_numpy_search(arr, targets)
    t_bin, r_bin = benchmark_parallel_binary_search(arr, targets, num_workers=256)
    t_int, r_int = benchmark_parallel_interpolation_search(arr, targets, num_workers=256)
    t_fib, r_fib = benchmark_parallel_fibonacci_search(arr, targets, num_workers=256)
    
    # 4) Validation
    print("Validating results...")
    mismatches_hybm = validate_results(res_hybm, res_np, targets, arr)
    mismatches_bin = validate_results(r_bin, res_np, targets, arr)
    mismatches_int = validate_results(r_int, res_np, targets, arr)
    mismatches_fib = validate_results(r_fib, res_np, targets, arr)
    
    # 5) Calculate performance metrics
    times = {
        'HyBMSearch': float(t_hybm),
        'NumPy_searchsorted': float(t_np),
        'Parallel_Binary_Py': float(t_bin),
        'Parallel_Interpolation_Py': float(t_int),
        'Parallel_Fibonacci_Py': float(t_fib),
    }
    
    speedups = {
        'vs_NumPy': float(t_np / t_hybm),
        'vs_Parallel_Binary': float(t_bin / t_hybm),
        'vs_Parallel_Interpolation': float(t_int / t_hybm),
        'vs_Parallel_Fibonacci': float(t_fib / t_hybm),
    }
    
    # 6) Save results
    results = {
        'distribution': distribution_type,
        'times': times,
        'speedups': speedups,
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
        'optimization_time': float(opt_time),
        'memory_usage_gb': float(arr.nbytes / (1024**3)),
    }
    
    with open(os.path.join(outdir, 'benchmarks', 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 7) Print summary
    print(f"\n--- {distribution_type.upper()} Results ---")
    print(f"HyBMSearch: {t_hybm:.2f}s")
    print(f"NumPy: {t_np:.2f}s (speedup: {speedups['vs_NumPy']:.1f}x)")
    print(f"Parallel Binary: {t_bin:.2f}s (speedup: {speedups['vs_Parallel_Binary']:.1f}x)")
    print(f"Parallel Interpolation: {t_int:.2f}s (speedup: {speedups['vs_Parallel_Interpolation']:.1f}x)")
    print(f"Parallel Fibonacci: {t_fib:.2f}s (speedup: {speedups['vs_Parallel_Fibonacci']:.1f}x)")
    print(f"Mismatches: {mismatches_hybm}")
    print(f"Memory: {arr.nbytes / (1024**3):.2f} GB")
    
    return results


def main():
    """Main benchmark runner."""
    # Configure logging
    setup_logger("benchmark_1b_int32.log")
    
    # Benchmark configuration
    SIZE = 1_000_000_000  # 1 billion elements
    TARGET_COUNT = 100_000_000  # 100M targets (10% of array size)
    POP_SIZE = 30  # Smaller population for faster optimization
    NGEN = 8  # Fewer generations for faster optimization
    
    # Data distributions to test
    DISTRIBUTIONS = [
        "sequential",      # Perfect sequential: 0, 1, 2, ...
        "random_uniform",  # Random uniform distribution
        "sparse",          # Sparse data with large gaps
        "exponential",     # Exponential distribution
        "bimodal",         # Bimodal distribution (two peaks)
    ]
    
    # Create output directory
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    base_outdir = os.path.join(PROJECT_ROOT, f"benchmark_1b_int32_{timestamp}")
    ensure_dir(base_outdir)
    
    print(f"=== 1 Billion Element Benchmark with int32 ===")
    print(f"Array size: {SIZE:,}")
    print(f"Target count: {TARGET_COUNT:,}")
    print(f"Memory per array: {SIZE * 4 / (1024**3):.2f} GB")
    print(f"Distributions: {', '.join(DISTRIBUTIONS)}")
    print(f"Output directory: {base_outdir}")
    
    # Run benchmarks for each distribution
    all_results = {}
    total_start = time.time()
    
    for dist_type in DISTRIBUTIONS:
        try:
            results = run_benchmark_for_distribution(
                dist_type, SIZE, TARGET_COUNT, POP_SIZE, NGEN, base_outdir
            )
            all_results[dist_type] = results
        except Exception as e:
            print(f"Error benchmarking {dist_type}: {e}")
            all_results[dist_type] = {"error": str(e)}
    
    total_time = time.time() - total_start
    
    # Save global summary
    summary = {
        'configuration': {
            'size': SIZE,
            'target_count': TARGET_COUNT,
            'pop_size': POP_SIZE,
            'ngen': NGEN,
            'dtype': 'int32',
            'memory_per_array_gb': SIZE * 4 / (1024**3),
        },
        'distributions': DISTRIBUTIONS,
        'total_time_seconds': float(total_time),
        'results': all_results,
    }
    
    with open(os.path.join(base_outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total benchmark time: {total_time:.2f}s")
    print(f"Results saved to: {base_outdir}")
    
    # Performance summary table
    print(f"\n--- Performance Summary ---")
    print(f"{'Distribution':<15} {'HyBMSearch':<12} {'NumPy':<12} {'Speedup':<10} {'Mismatches':<12}")
    print("-" * 70)
    
    for dist_type, results in all_results.items():
        if 'error' not in results:
            hybm_time = results['times']['HyBMSearch']
            numpy_time = results['times']['NumPy_searchsorted']
            speedup = results['speedups']['vs_NumPy']
            mismatches = results['mismatches']['HyBMSearch']
            print(f"{dist_type:<15} {hybm_time:<12.2f} {numpy_time:<12.2f} {speedup:<10.1f}x {mismatches:<12}")
        else:
            print(f"{dist_type:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<12}")


if __name__ == '__main__':
    main()
