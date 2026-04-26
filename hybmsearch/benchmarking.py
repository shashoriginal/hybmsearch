"""
Benchmarking and Validation

Performance testing and result validation functionality.
Includes benchmarks against NumPy and standard parallel algorithms.
"""

import time
import logging
from typing import Tuple, Optional
import multiprocessing

import numpy as np

from .core import perform_search, SearchConfig
from .algorithms.standard import (
    parallel_binary_search,
    parallel_interpolation_search,
    parallel_fibonacci_search
)


logger = logging.getLogger(__name__)


def benchmark_search(arr: np.ndarray, targets: np.ndarray, 
                    config: Optional[SearchConfig] = None,
                    generation: int = 0, **kwargs) -> Tuple[float, np.ndarray]:
    """
    Benchmark the HyBMSearch algorithm with given configuration.
    
    Args:
        arr: Sorted array to search in
        targets: Target values to search for
        config: SearchConfig object (optional)
        generation: Generation number (for compatibility)
        **kwargs: Alternative way to specify config parameters
        
    Returns:
        Tuple of (elapsed_time, results_array)
    """
    results, elapsed = perform_search(arr, targets, generation, config, **kwargs)
    return elapsed, results


def benchmark_numpy_search(arr: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Benchmark NumPy's searchsorted as baseline.
    
    Args:
        arr: Sorted array to search in
        targets: Target values to search for
        
    Returns:
        Tuple of (elapsed_time, results_array)
    """
    start = time.time()
    # Use searchsorted to find insertion points
    indices = np.searchsorted(arr, targets, side='left') # 'left' finds first occurrence potential index

    # Validate the found indices
    # Handle indices out of bounds (target > max(arr))
    indices_in_bounds = indices < arr.size
    valid_indices = indices[indices_in_bounds]

    # Check if the element at the found index actually matches the target
    found_mask = np.zeros_like(targets, dtype=bool)
    if valid_indices.size > 0: # Only check if there are any in-bounds indices
         match_mask = arr[valid_indices] == targets[indices_in_bounds]
         found_mask[indices_in_bounds] = match_mask

    # Initialize results to -1 (not found)
    results = np.full(targets.shape, -1, dtype=np.int64)
    # Update results with the index where a match was found
    results[found_mask] = indices[found_mask]

    elapsed = time.time() - start
    return elapsed, results


def benchmark_parallel_binary_search(arr: np.ndarray, targets: np.ndarray, 
                                    num_workers: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Benchmark parallel binary search using ThreadPoolExecutor.
    
    Args:
        arr: Sorted array to search in
        targets: Target values to search for
        num_workers: Number of worker threads
        
    Returns:
        Tuple of (elapsed_time, results_array)
    """
    logger.info(f"Starting Parallel Binary Search Benchmark (ThreadPoolExecutor, {num_workers or multiprocessing.cpu_count()} workers)...")
    start = time.time()
    result = parallel_binary_search(arr, targets, num_workers)
    elapsed = time.time() - start
    logger.info("Finished Parallel Binary Search Benchmark.")
    return elapsed, result


def benchmark_parallel_interpolation_search(arr: np.ndarray, targets: np.ndarray,
                                           num_workers: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Benchmark parallel interpolation search using ThreadPoolExecutor.
    
    Args:
        arr: Sorted array to search in
        targets: Target values to search for
        num_workers: Number of worker threads
        
    Returns:
        Tuple of (elapsed_time, results_array)
    """
    logger.info(f"Starting Parallel Interpolation Search Benchmark (ThreadPoolExecutor, {num_workers or multiprocessing.cpu_count()} workers)...")
    start = time.time()
    result = parallel_interpolation_search(arr, targets, num_workers)
    elapsed = time.time() - start
    logger.info("Finished Parallel Interpolation Search Benchmark.")
    return elapsed, result


def benchmark_parallel_fibonacci_search(arr: np.ndarray, targets: np.ndarray,
                                       num_workers: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Benchmark parallel Fibonacci search using ThreadPoolExecutor.
    
    Args:
        arr: Sorted array to search in
        targets: Target values to search for
        num_workers: Number of worker threads
        
    Returns:
        Tuple of (elapsed_time, results_array)
    """
    logger.info(f"Starting Parallel Fibonacci Search Benchmark (ThreadPoolExecutor, {num_workers or multiprocessing.cpu_count()} workers)...")
    start = time.time()
    result = parallel_fibonacci_search(arr, targets, num_workers)
    elapsed = time.time() - start
    logger.info("Finished Parallel Fibonacci Search Benchmark.")
    return elapsed, result


def validate_results(results_ours: np.ndarray, results_baseline: np.ndarray, 
                    targets: np.ndarray, arr: np.ndarray = None) -> int:
    """
    Compare results from our search method against baseline results.
    Handles duplicates correctly by checking if both results point to the same value.
    
    Args:
        results_ours: Results from HyBMSearch
        results_baseline: Results from baseline method (e.g., NumPy)
        targets: Target values that were searched for
        arr: The array that was searched (needed for duplicate validation)
        
    Returns:
        int: Number of actual mismatches (0 means perfect match)
    """
    if results_ours.shape != results_baseline.shape:
        logger.error(f"Result shapes mismatch! Ours: {results_ours.shape}, Baseline: {results_baseline.shape}")
        return -1 # Indicate error

    total_targets = targets.size
    
    # Count how many targets were found by each method
    # Found means result is not -1
    found_ours = np.sum(results_ours != -1)
    found_baseline = np.sum(results_baseline != -1)

    # Check for actual mismatches (considering duplicates)
    actual_mismatches = 0
    mismatch_details = []
    
    for i in range(total_targets):
        our_idx = results_ours[i]
        baseline_idx = results_baseline[i]
        target = targets[i]
        
        if our_idx != baseline_idx:
            # Indices differ, but check if both are valid for the same target
            if our_idx == -1 and baseline_idx == -1:
                # Both not found - this is correct
                continue
            elif our_idx == -1 or baseline_idx == -1:
                # One found, one not found - this is a real mismatch
                actual_mismatches += 1
                mismatch_details.append((i, target, our_idx, baseline_idx, "found_vs_not_found"))
            elif arr is not None:
                # Both found something - check if they point to the same value
                if our_idx < len(arr) and baseline_idx < len(arr):
                    if arr[our_idx] == target and arr[baseline_idx] == target:
                        # Both found valid occurrences of the same target (duplicates case)
                        continue
                    else:
                        # At least one points to wrong value
                        actual_mismatches += 1
                        mismatch_details.append((i, target, our_idx, baseline_idx, "wrong_value"))
                else:
                    # Index out of bounds
                    actual_mismatches += 1  
                    mismatch_details.append((i, target, our_idx, baseline_idx, "out_of_bounds"))
            else:
                # No array provided, treat as mismatch
                actual_mismatches += 1
                mismatch_details.append((i, target, our_idx, baseline_idx, "no_array_check"))

    logger.info(f"Validation vs Baseline:")
    logger.info(f"  Targets searched: {total_targets}")
    logger.info(f"  Targets found (Our method): {found_ours}")
    logger.info(f"  Targets found (Baseline):   {found_baseline}")
    logger.info(f"  Index differences: {np.sum(results_ours != results_baseline)}")
    logger.info(f"  Actual mismatches: {actual_mismatches}")

    if actual_mismatches > 0:
         # Log some specific mismatches for debugging
         num_to_show = min(5, len(mismatch_details))
         logger.warning(f"  First {num_to_show} actual mismatches:")
         for i in range(num_to_show):
             idx, target, our_idx, baseline_idx, reason = mismatch_details[i]
             logger.warning(f"    Target {target} -> Ours: {our_idx}, Baseline: {baseline_idx} ({reason})")
         return actual_mismatches
    else:
        logger.info("  Results are equivalent (handling duplicates correctly).")
        return 0


def run_comprehensive_benchmark(arr: np.ndarray, targets: np.ndarray,
                               config: Optional[SearchConfig] = None) -> dict:
    """
    Run a comprehensive benchmark comparing HyBMSearch against all baselines.
    
    Args:
        arr: Sorted array to search in
        targets: Target values to search for
        config: SearchConfig for HyBMSearch (if None, uses defaults)
        
    Returns:
        dict: Benchmark results with timing and validation info
    """
    logger.info("="*70)
    logger.info("Running Comprehensive Benchmark")
    logger.info("="*70)
    
    results = {}
    
    # Benchmark HyBMSearch with given config
    logger.info("--- Benchmarking HyBMSearch ---")
    t_ours, res_ours = benchmark_search(arr, targets, config=config)
    results['hybmsearch'] = {'time': t_ours, 'results': res_ours}
    logger.info(f"HyBMSearch Time: {t_ours:.6f}s")
    
    # Benchmark NumPy baseline
    logger.info("--- Benchmarking NumPy searchsorted (Baseline) ---")
    t_np, res_np = benchmark_numpy_search(arr, targets)
    results['numpy'] = {'time': t_np, 'results': res_np}
    logger.info(f"NumPy searchsorted Time: {t_np:.6f}s")
    
    # Validate against NumPy
    logger.info("--- Validating HyBMSearch vs NumPy ---")
    mismatches = validate_results(res_ours, res_np, targets, arr)
    results['validation'] = {'mismatches': mismatches}
    
    # Benchmark standard parallel algorithms
    num_workers = multiprocessing.cpu_count()
    logger.info(f"--- Benchmarking Standard Parallel Algorithms ({num_workers} workers) ---")
    
    # Parallel Binary Search
    t_pbs, r_pbs = benchmark_parallel_binary_search(arr, targets, num_workers=num_workers)
    results['parallel_binary'] = {'time': t_pbs, 'results': r_pbs}
    logger.info(f"[Parallel Binary Search] Time = {t_pbs:.6f}s")
    validate_results(r_pbs, res_np, targets, arr)
    
    # Parallel Interpolation Search
    t_pis, r_pis = benchmark_parallel_interpolation_search(arr, targets, num_workers=num_workers)
    results['parallel_interpolation'] = {'time': t_pis, 'results': r_pis}
    logger.info(f"[Parallel Interpolation Search] Time = {t_pis:.6f}s")
    validate_results(r_pis, res_np, targets, arr)
    
    # Parallel Fibonacci Search
    t_fib, r_fib = benchmark_parallel_fibonacci_search(arr, targets, num_workers=num_workers)
    results['parallel_fibonacci'] = {'time': t_fib, 'results': r_fib}
    logger.info(f"[Parallel Fibonacci Search] Time = {t_fib:.6f}s")
    validate_results(r_fib, res_np, targets, arr)
    
    # Summary
    logger.info("-" * 70)
    logger.info(f"Benchmark Summary for Array Size = {arr.size:,}")
    logger.info(f"  HyBMSearch:                 {t_ours:.6f}s")
    logger.info(f"  NumPy searchsorted:         {t_np:.6f}s")
    logger.info(f"  Parallel Binary:            {t_pbs:.6f}s")
    logger.info(f"  Parallel Interpolation:     {t_pis:.6f}s")
    logger.info(f"  Parallel Fibonacci:         {t_fib:.6f}s")
    logger.info("-" * 70)
    
    return results
