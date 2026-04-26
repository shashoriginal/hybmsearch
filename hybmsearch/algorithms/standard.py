"""
Standard Parallel Search Algorithms

ThreadPoolExecutor-based implementations of standard search algorithms
for comparison and benchmarking purposes.
"""

import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


def _py_binary_search(arr, t):
    """Standard iterative binary search in Python."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) >> 1 # Faster integer division for midpoint
        if mid < 0 or mid >= len(arr): # Basic bounds check
            return -1 # Should not happen with correct logic
        val = arr[mid]
        if val == t:
            return mid
        elif val < t:
            left = mid + 1
        else:
            right = mid - 1
    return -1 # Target not found


def _py_interpolation_search(arr, t):
    """Standard iterative interpolation search in Python."""
    left, right = 0, len(arr) - 1
    n = len(arr)

    # Check initial bounds
    if n == 0 or t < arr[left] or t > arr[right]:
         return -1

    while left <= right and t >= arr[left] and t <= arr[right]:
        if arr[right] == arr[left]:
            return left if arr[left] == t else -1

        # Calculate position
        # Ensure denominator is non-zero (checked above)
        # Use floating point arithmetic
        pos = left + int( (float(right - left) / (arr[right] - arr[left])) * (t - arr[left]) )

        # Boundary check for calculated position
        if pos < left or pos > right:
             # Fallback to binary search within the current [left, right] range
             # This makes it more robust if interpolation jumps out of bounds
             return _py_binary_search_slice(arr, t, left, right)

        val = arr[pos]
        if val == t:
            return pos
        elif val < t:
            left = pos + 1
        else:
            right = pos - 1

    # Check boundary case after loop (e.g., if loop condition breaks early)
    if left < n and arr[left] == t:
        return left

    return -1


def _py_binary_search_slice(arr, t, left, right):
    """Helper for fallback in interpolation search."""
    while left <= right:
        mid = (left + right) >> 1
        if mid < 0 or mid >= len(arr): return -1 # Bounds check
        val = arr[mid]
        if val == t: return mid
        elif val < t: left = mid + 1
        else: right = mid - 1
    return -1


def _py_fibonacci_search(arr, t):
    """
    A standard Fibonacci Search for a single target 't' in sorted 'arr'.
    Returns the index or -1 if not found.
    """
    n = len(arr)
    if n == 0: return -1 # Handle empty array

    # Initialize Fibonacci numbers
    fibM2 = 0  # (m-2)'th Fibonacci No.
    fibM1 = 1  # (m-1)'th Fibonacci No.
    fibM = fibM2 + fibM1  # m'th Fibonacci

    # Find the smallest Fibonacci number greater than or equal to n
    while fibM < n:
        fibM2 = fibM1
        fibM1 = fibM
        fibM = fibM2 + fibM1

    offset = -1 # Marks the eliminated range from front

    # While there are elements to be inspected.
    # Note that we compare arr[fibM2 + offset] with t.
    while fibM > 1:
        # Check if fibM2 is a valid index
        i = min(offset + fibM2, n - 1)

        # If t is greater than the value at index fibM2,
        # cut the subarray array from offset to i
        if arr[i] < t:
            fibM = fibM1
            fibM1 = fibM2
            fibM2 = fibM - fibM1
            offset = i

        # If t is less than the value at index fibM2,
        # cut the subarray after i+1
        elif arr[i] > t:
            fibM = fibM2
            fibM1 = fibM1 - fibM2
            fibM2 = fibM - fibM1

        # element found. return index
        else:
            return i

    # Compare the last element with t
    # Need to check if fibM1 is 1 and the last comparison element index (offset+1) is valid
    if fibM1 == 1 and offset + 1 < n and arr[offset + 1] == t:
        return offset + 1

    # element not found. return -1
    return -1


def _worker_search(arr, targets, start_idx, end_idx, search_func):
    """Helper function to perform search on a slice of targets."""
    sub_t = targets[start_idx:end_idx]
    sub_out = np.empty(len(sub_t), dtype=np.int64)
    for i, val in enumerate(sub_t):
        sub_out[i] = search_func(arr, val)
    return (start_idx, sub_out)


def parallel_binary_search(arr, targets, num_workers=None):
    """
    Parallel binary search using ThreadPoolExecutor.
    
    Args:
        arr: Sorted array to search in
        targets: Array of target values
        num_workers: Number of worker threads (default: CPU count)
        
    Returns:
        numpy.ndarray: Array of indices (-1 if not found)
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, num_workers) # Ensure at least one worker

    out = np.empty(len(targets), dtype=np.int64)
    # Determine chunk size for task distribution
    total_targets = len(targets)
    # Aim for a reasonable number of tasks, e.g., 4 per worker, or ensure chunk size > 0
    chunk_size = max(1, total_targets // (num_workers * 4))

    futures = []
    # Using ThreadPoolExecutor for I/O bound or GIL-releasing tasks (like calling Numba/C code)
    # Pure Python compute tasks might be better with ProcessPoolExecutor, but ThreadPool
    # is often used here assuming the underlying search could release GIL or is external.
    # Given _py_binary_search is pure Python, ProcessPoolExecutor might be faster if GIL is limiting.
    # Let's stick to ThreadPoolExecutor as per the original code structure.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        idx = 0
        while idx < total_targets:
            end_idx = min(idx + chunk_size, total_targets)
            # Submit a task to search the sub-array targets[idx:end_idx]
            futures.append(executor.submit(_worker_search, arr, targets, idx, end_idx, _py_binary_search))
            idx = end_idx

        # Collect results as they complete
        for f in futures:
            start_idx, sub_res = f.result()
            out[start_idx : start_idx + len(sub_res)] = sub_res

    return out


def parallel_interpolation_search(arr, targets, num_workers=None):
    """
    Parallel interpolation search using ThreadPoolExecutor.
    
    Args:
        arr: Sorted array to search in
        targets: Array of target values
        num_workers: Number of worker threads (default: CPU count)
        
    Returns:
        numpy.ndarray: Array of indices (-1 if not found)
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, num_workers)

    out = np.empty(len(targets), dtype=np.int64)
    total_targets = len(targets)
    chunk_size = max(1, total_targets // (num_workers * 4))

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        idx = 0
        while idx < total_targets:
            end_idx = min(idx + chunk_size, total_targets)
            futures.append(executor.submit(_worker_search, arr, targets, idx, end_idx, _py_interpolation_search))
            idx = end_idx

        for f in futures:
            start_idx, sub_res = f.result()
            out[start_idx : start_idx + len(sub_res)] = sub_res

    return out


def parallel_fibonacci_search(arr, targets, num_workers=None):
    """
    Parallel Fibonacci search using ThreadPoolExecutor.
    
    Args:
        arr: Sorted array to search in
        targets: Array of target values
        num_workers: Number of worker threads (default: CPU count)
        
    Returns:
        numpy.ndarray: Array of indices (-1 if not found)
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, num_workers)

    out = np.empty(len(targets), dtype=np.int64)
    total_targets = len(targets)
    chunk_size = max(1, total_targets // (num_workers * 4))

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        idx = 0
        while idx < total_targets:
            end_idx = min(idx + chunk_size, total_targets)
            # Submit tasks using the _worker_search helper
            futures.append(executor.submit(_worker_search, arr, targets, idx, end_idx, _py_fibonacci_search))
            idx = end_idx

        # Collect results
        for f in futures:
            start_idx, sub_res = f.result()
            out[start_idx : start_idx + len(sub_res)] = sub_res

    return out
