"""
Utilities

Common utility functions for logging, threading, and configuration.
"""

import logging
import multiprocessing
import warnings
from typing import Optional

from numba import set_num_threads, config


# Configure warnings and Numba settings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Choose the threading layer
config.THREADING_LAYER = "workqueue"  # or "omp" if preferred/available


def setup_logger(log_file: str = "hybmsearch_results.txt", 
                level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger for HyBMSearch.
    
    Args:
        log_file: File to write logs to
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("hybmsearch")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def set_optimal_num_threads(num_threads: Optional[int] = None) -> None:
    """
    Set the optimal number of threads for Numba parallel operations.
    
    Args:
        num_threads: Desired number of threads (default: CPU count)
    """
    if num_threads is None:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = min(num_threads, multiprocessing.cpu_count())
    
    set_num_threads(num_cores)
    
    # Log the setting
    logger = logging.getLogger("hybmsearch")
    try:
        current_threads = getattr(config, 'NUMBA_NUM_THREADS', num_cores)
        if current_threads != num_cores:
            logger.info(f"Setting Numba threads to {num_cores}.")
            set_num_threads(num_cores)
    except AttributeError:
        logger.info(f"Setting Numba threads for the first time to {num_cores}.")
        set_num_threads(num_cores)


def get_system_info() -> dict:
    """
    Get system information for benchmarking context.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'cpu_count': multiprocessing.cpu_count(),
        'numba_threading_layer': config.THREADING_LAYER,
    }
    
    try:
        info['numba_threads'] = getattr(config, 'NUMBA_NUM_THREADS', 'unknown')
    except AttributeError:
        info['numba_threads'] = 'not_set'
    
    return info


def validate_array_inputs(arr, targets):
    """
    Validate input arrays for search operations.
    
    Args:
        arr: Main array to search in
        targets: Target values to search for
        
    Raises:
        ValueError: If inputs are invalid
        TypeError: If inputs are wrong type
    """
    import numpy as np
    
    # Type checks
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy array")
    if not isinstance(targets, np.ndarray):
        raise TypeError("targets must be a numpy array")
    
    # Shape checks
    if arr.ndim != 1:
        raise ValueError("arr must be 1-dimensional")
    if targets.ndim != 1:
        raise ValueError("targets must be 1-dimensional")
    
    # Size checks
    if arr.size == 0:
        raise ValueError("arr cannot be empty")
    if targets.size == 0:
        raise ValueError("targets cannot be empty")
    
    # Data type compatibility
    if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.floating):
        raise ValueError("arr must contain numeric data")
    if not np.issubdtype(targets.dtype, np.integer) and not np.issubdtype(targets.dtype, np.floating):
        raise ValueError("targets must contain numeric data")
    
    # Sortedness check (basic)
    if arr.size > 1 and not np.all(arr[:-1] <= arr[1:]):
        raise ValueError("arr must be sorted in non-decreasing order")


def create_test_data(size: int, target_count: Optional[int] = None, 
                    data_type: str = "sequential", dtype=None):
    """
    Create test data for benchmarking.
    
    Args:
        size: Size of the main array
        target_count: Number of targets (default: size // 100)
        data_type: Type of data distribution ('sequential', 'random', 'sparse')
        dtype: NumPy data type (default: np.int64)
        
    Returns:
        Tuple of (array, targets)
    """
    import numpy as np
    
    if dtype is None:
        dtype = np.int64
    
    if target_count is None:
        target_count = max(1000, size // 100)
    
    # Create main array based on data type
    if data_type == "sequential":
        arr = np.arange(size, dtype=dtype)
    elif data_type == "random":
        arr = np.sort(np.random.randint(0, size * 2, size=size, dtype=dtype))
    elif data_type == "sparse":
        arr = np.sort(np.random.randint(0, size * 10, size=size, dtype=dtype))
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Create targets - mix of found and not-found
    min_val, max_val = arr[0], arr[-1]
    targets_found = np.random.choice(arr, size=target_count // 2, replace=True)
    targets_random = np.random.randint(min_val, max_val + 1, size=target_count - len(targets_found), dtype=dtype)
    targets = np.concatenate([targets_found, targets_random])
    np.random.shuffle(targets)
    
    return arr, targets
