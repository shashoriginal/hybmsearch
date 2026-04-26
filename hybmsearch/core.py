"""
Core Search Interface

Main search functionality and configuration for HyBMSearch.
Provides the unified interface for all search strategies.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from .algorithms.parallel import (
    parallel_chunk_search,
    parallel_twolevel_search,
    parallel_vector_pivot_search,
    merge_search
)
from .utils import set_optimal_num_threads


@dataclass
class SearchConfig:
    """
    Configuration class for search parameters.
    
    Optimized for large-scale datasets with intelligent defaults
    designed to outperform NumPy across all data distributions.
    
    Attributes:
        use_merge_search: Use merge search for sorted targets
        num_levels: Number of chunk levels (1 or 2)
        chunk_size: Size of chunks for chunking algorithms
        sub_chunk_size: Size of sub-chunks for two-level search
        use_interpolation: Use interpolation search within chunks
        num_threads: Number of threads for parallel processing
        use_vector_pivot: Use vector pivot search algorithm
        pivot_count: Number of pivots for VPS algorithm
    """
    use_merge_search: bool = False
    num_levels: int = 1
    chunk_size: int = 2048  # CPU cache-optimized for maximum speed
    sub_chunk_size: int = 512  # L1 cache-friendly
    use_interpolation: bool = False  # Binary search is faster for most cases
    num_threads: int = -1  # Auto-detect optimal thread count
    use_vector_pivot: bool = False  # Simplified for speed
    pivot_count: int = 32  # Reasonable default


def perform_search(
    arr: np.ndarray,
    targets: np.ndarray,
    generation: int = 0,
    config: Optional[SearchConfig] = None,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Perform search using the specified strategy and configuration.
    
    Args:
        arr: Sorted numpy array to search in
        targets: Array of target values to search for  
        generation: Generation number (for compatibility, not used)
        config: SearchConfig object with parameters
        **kwargs: Alternative way to specify config parameters
        
    Returns:
        Tuple of (results_array, elapsed_time)
        
    Examples:
        >>> arr = np.arange(1000000)
        >>> targets = np.random.randint(0, 1000000, 10000)
        >>> config = SearchConfig(use_vector_pivot=True, pivot_count=32)
        >>> results, time_taken = perform_search(arr, targets, config=config)
        
        >>> # Alternative using kwargs
        >>> results, time_taken = perform_search(
        ...     arr, targets, 
        ...     use_merge_search=True, 
        ...     num_threads=4
        ... )
    """
    # Handle configuration
    if config is None:
        config = SearchConfig(**kwargs)
    else:
        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Auto-detect optimal thread count for large datasets
    if config.num_threads == -1:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Scale thread count with data size: more data = more threads
        if arr.size > 100_000_000:  # 100M+ elements
            config.num_threads = cpu_count
        elif arr.size > 10_000_000:  # 10M+ elements
            config.num_threads = max(cpu_count // 2, 4)
        else:  # Smaller datasets
            config.num_threads = max(cpu_count // 4, 2)
    
    # Adaptive chunk sizing for optimal performance across scales
    if config.chunk_size == 2048:  # Using default
        if arr.size > 1_000_000_000:  # 1B+ elements: larger chunks for efficiency
            config.chunk_size = min(65536, arr.size // (config.num_threads * 8))
        elif arr.size > 100_000_000:  # 100M+ elements: balanced chunks
            config.chunk_size = min(16384, arr.size // (config.num_threads * 16))
        elif arr.size > 10_000_000:  # 10M+ elements: cache-friendly chunks
            config.chunk_size = min(8192, arr.size // (config.num_threads * 32))
        # else: keep default 2048 for smaller arrays
        
        # Ensure minimum chunk size for meaningful parallelization
        config.chunk_size = max(config.chunk_size, 512)
    
    set_optimal_num_threads(config.num_threads)
    n = arr.size

    # Proceed with the chosen search strategy
    start = time.time()

    if config.use_merge_search:
        # Merge search requires targets to be sorted first
        sorted_indices = np.argsort(targets)
        sorted_targets = targets[sorted_indices]
        # Allocate temporary results array for merge_search output
        tmp_results = np.empty_like(sorted_targets, dtype=np.int64)
        merge_search(arr, sorted_targets, tmp_results)
        # Re-order results back to the original target order
        results = np.empty_like(tmp_results)
        results[sorted_indices] = tmp_results
        elapsed = time.time() - start
        return results, elapsed

    # If not using merge search, allocate results array here
    results = np.empty(targets.size, dtype=np.int64)

    if config.use_vector_pivot:
        # Vector Pivot Search uses its own parallel Numba implementation
        parallel_vector_pivot_search(arr, targets, results, config.pivot_count)
        elapsed = time.time() - start
        return results, elapsed

    # Default to Chunk-based approach if neither merge nor VPS is selected
    if config.num_levels == 1:
        parallel_chunk_search(arr, targets, results, config.chunk_size, config.use_interpolation)
    else: # Assuming num_levels == 2 or more defaults to two-level
        parallel_twolevel_search(
            arr, targets, results, 
            config.chunk_size, config.sub_chunk_size, 
            config.use_interpolation
        )

    elapsed = time.time() - start
    return results, elapsed
