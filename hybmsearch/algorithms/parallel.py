"""
Parallel Search Algorithms

Numba-optimized parallel search implementations including:
- Single-level chunk search
- Two-level hierarchical chunk search  
- Vector pivot search (VPS)
- Merge search for sorted targets
"""

import numpy as np
from numba import njit, prange
from .core import binary_search, interpolation_search


@njit(parallel=True, fastmath=True, cache=True)
def parallel_chunk_search(arr, targets, results, chunk_size, use_interpolation=False):
    """
    Single-level parallel chunk search with robust chunk selection.
    
    Uses binary search on chunk boundaries to find the correct chunk for each target,
    making it robust for any data distribution.
    
    Args:
        arr: Sorted numpy array to search in
        targets: Array of target values to search for
        results: Output array for results (modified in-place)
        chunk_size: Size of each chunk
        use_interpolation: Whether to use interpolation search within chunks
    """
    n = arr.size
    if n == 0: # Handle empty array case
        results[:] = -1
        return
    num_targets = targets.size
    
    # Smart chunk sizing: disable chunking for small arrays, use reasonable sizes for large ones
    if n < 1000:
        # For small arrays, use single chunk (no chunking overhead)
        effective_chunk_size = n
    else:
        # For larger arrays, target 8-16 chunks for optimal balance
        optimal_chunks = max(8, min(16, n // 5000))
        adaptive_chunk_size = max(100, n // optimal_chunks)  # Minimum 100 elements per chunk
        effective_chunk_size = min(chunk_size, adaptive_chunk_size)
    
    num_chunks = max(1, (n + effective_chunk_size - 1) // effective_chunk_size)

    # Pre-compute chunk boundaries for efficient lookup
    chunk_boundaries = np.empty(num_chunks + 1, dtype=np.int64)
    for i in range(num_chunks + 1):
        chunk_boundaries[i] = min(i * effective_chunk_size, n)

    # Pre-calculate value range for chunk estimation
    max_val = arr[-1]
    min_val = arr[0]
    val_range = max_val - min_val
    if val_range <= 0:
        val_range = 1  # Avoid division by zero

    # Distribution uniformity check: if data is highly non-uniform, skip chunking
    use_chunking = True
    if num_chunks > 1 and n > 100:  # Only check for larger arrays with multiple chunks
        # Sample a few points to check uniformity
        sample_indices = [n // 4, n // 2, 3 * n // 4]
        expected_values = [min_val + (val_range * idx // (n - 1)) for idx in sample_indices]
        actual_values = [arr[idx] for idx in sample_indices]
        
        # Check if actual values deviate significantly from expected uniform distribution
        max_deviation = 0
        for expected, actual in zip(expected_values, actual_values):
            deviation = abs(actual - expected) / val_range if val_range > 0 else 0
            max_deviation = max(max_deviation, deviation)
        
        # If deviation is too high, disable chunking (use full binary search)
        if max_deviation > 0.3:  # 30% deviation threshold
            use_chunking = False

    for i in prange(num_targets):
        t = targets[i]

        if use_chunking and num_chunks > 1:
            # Original research algorithm: Estimate chunk index based on target value
            # This is the core innovation - intelligent chunk selection
            relative_pos = (float(t - min_val) / val_range) if val_range > 1 else 0.0
            relative_pos = max(0.0, min(1.0, relative_pos))  # Clamp to [0, 1]
            # Fix floating-point precision issue for large arrays
            estimated_idx = int(relative_pos * (n - 1))
            estimated_idx = max(0, min(estimated_idx, n - 1))  # Clamp to valid array bounds
            
            # Map estimated index to chunk index
            chunk_idx = estimated_idx // effective_chunk_size
            chunk_idx = max(0, min(chunk_idx, num_chunks - 1))  # Clamp chunk index
            
            # Define search bounds based on the estimated chunk
            left = chunk_boundaries[chunk_idx]
            right = chunk_boundaries[chunk_idx + 1] - 1
            right = min(right, n - 1)
            
            # Perform search within the estimated chunk
            if left <= right:
                if use_interpolation:
                    idx = interpolation_search(arr, t, left, right)
                else:
                    idx = binary_search(arr, t, left, right)
                
                if idx != -1:
                    results[i] = idx
                else:
                    # Ultimate fallback: if chunk estimation fails completely,
                    # fall back to standard binary search on the full array
                    results[i] = binary_search(arr, t, 0, n - 1)
            else:
                results[i] = -1
        else:
            # For non-uniform data or single chunk, use standard binary search
            results[i] = binary_search(arr, t, 0, n - 1)


@njit(parallel=True, fastmath=True, cache=True)
def parallel_twolevel_search(arr, targets, results, chunk_size, sub_chunk_size, use_interpolation=False):
    """
    Two-level hierarchical parallel chunk search with robust chunk selection.
    
    Uses binary search to find correct chunks at both levels, making it robust
    for any data distribution.
    
    Args:
        arr: Sorted numpy array to search in
        targets: Array of target values to search for
        results: Output array for results (modified in-place)
        chunk_size: Size of each top-level chunk
        sub_chunk_size: Size of each sub-chunk
        use_interpolation: Whether to use interpolation search within sub-chunks
    """
    n = arr.size
    if n == 0: # Handle empty array
        results[:] = -1
        return
    num_targets = targets.size
    num_top_chunks = max(1, (n + chunk_size - 1) // chunk_size) # Ensure at least 1 chunk

    # Pre-compute top-level chunk boundaries
    top_chunk_boundaries = np.empty(num_top_chunks + 1, dtype=np.int64)
    for i in range(num_top_chunks + 1):
        top_chunk_boundaries[i] = min(i * chunk_size, n)

    # Pre-calculate value range for estimation
    max_val = arr[-1]
    min_val = arr[0]
    val_range = max_val - min_val
    if val_range <= 0:
        val_range = 1

    # Distribution uniformity check: if data is highly non-uniform, skip chunking
    use_chunking = True
    if num_top_chunks > 1 and n > 100:  # Only check for larger arrays with multiple chunks
        # Sample a few points to check uniformity
        sample_indices = [n // 4, n // 2, 3 * n // 4]
        expected_values = [min_val + (val_range * idx // (n - 1)) for idx in sample_indices]
        actual_values = [arr[idx] for idx in sample_indices]
        
        # Check if actual values deviate significantly from expected uniform distribution
        max_deviation = 0
        for expected, actual in zip(expected_values, actual_values):
            deviation = abs(actual - expected) / val_range if val_range > 0 else 0
            max_deviation = max(max_deviation, deviation)
        
        # If deviation is too high, disable chunking (use full binary search)
        if max_deviation > 0.3:  # 30% deviation threshold
            use_chunking = False

    for i in prange(num_targets):
        t = targets[i]

        if use_chunking and num_top_chunks > 1:
            # Original research algorithm: Two-level intelligent estimation
            # Estimate top-level chunk index based on target value
            relative_pos = (float(t - min_val) / val_range) if val_range > 1 else 0.0
            relative_pos = max(0.0, min(1.0, relative_pos))
            # Fix floating-point precision issue for large arrays
            estimated_idx = int(relative_pos * (n - 1))
            estimated_idx = max(0, min(estimated_idx, n - 1))  # Clamp to valid array bounds

            top_idx = estimated_idx // chunk_size
            top_idx = max(0, min(top_idx, num_top_chunks - 1))

            top_left = top_chunk_boundaries[top_idx]
            top_right = top_chunk_boundaries[top_idx + 1] - 1
            top_right = min(top_right, n - 1)

            found = False
            if top_left <= top_right:
                # Estimate sub-chunk index within the top chunk
                relative_idx_in_chunk = estimated_idx - top_left
                relative_idx_in_chunk = max(0, relative_idx_in_chunk)
                
                current_chunk_size = top_right - top_left + 1
                num_sub_chunks = max(1, (current_chunk_size + sub_chunk_size - 1) // sub_chunk_size)
                
                sub_idx = relative_idx_in_chunk // sub_chunk_size
                sub_idx = max(0, min(sub_idx, num_sub_chunks - 1))
                
                # Calculate sub-chunk boundaries
                sub_left = top_left + sub_idx * sub_chunk_size
                sub_right = min(sub_left + sub_chunk_size - 1, top_right)
                
                # Ensure sub-level bounds are valid
                sub_left = max(top_left, sub_left)
                sub_right = min(top_right, sub_right)
                
                if sub_left <= sub_right:
                    # Perform search in the estimated sub-chunk
                    if use_interpolation:
                        idx = interpolation_search(arr, t, sub_left, sub_right)
                    else:
                        idx = binary_search(arr, t, sub_left, sub_right)
                    
                    if idx != -1:
                        results[i] = idx
                        found = True
            
            if not found:
                # Fallback: if two-level estimation fails, use full binary search
                results[i] = binary_search(arr, t, 0, n - 1)
        else:
            # For non-uniform data, use standard binary search
            results[i] = binary_search(arr, t, 0, n - 1)


@njit
def merge_search(arr, sorted_targets, indices_out):
    """
    Single-pass merge search for sorted targets.
    
    Performs a linear merge between the sorted array and sorted targets,
    achieving O(n + m) time complexity where n = array size, m = target count.
    
    Args:
        arr: Sorted numpy array to search in
        sorted_targets: Sorted array of target values
        indices_out: Output array for indices (modified in-place)
    """
    n = arr.size
    q = sorted_targets.size
    i = 0 # index for arr
    j = 0 # index for sorted_targets

    # Handle empty arrays
    if n == 0 or q == 0:
        indices_out[:] = -1 # If either array is empty, no matches possible
        return

    while i < n and j < q:
        a_val = arr[i]
        t_val = sorted_targets[j]

        if a_val < t_val:
            i += 1 # Move pointer in arr forward
        elif a_val > t_val:
            # Target t_val is not in arr at or after current position i
            # Since arr is sorted, it won't be found later either.
            indices_out[j] = -1
            j += 1 # Move to the next target
        else: # a_val == t_val
            # Found the target
            indices_out[j] = i
            j += 1 # Move to the next target
            # Optional: If duplicates in arr are possible and you need the *first* occurrence,
            # you might not advance 'i' immediately, or handle it differently.
            # Assuming we want *an* index if found, advancing 'i' is fine if duplicates in 'arr'
            # should map to the same index 'i' until arr[i] changes.
            # However, standard merge implies moving both pointers on match.
            # If we need to find multiple targets mapping to the same arr value,
            # we should only advance j until t_val changes.
            # Let's refine: Hold 'i' if the next target is the same.
            # This requires looking ahead in sorted_targets or a slightly different loop structure.

            # Simpler: Assume standard merge where we advance both on match.
            # If multiple targets match arr[i], subsequent ones will be handled in next iterations.
            # But wait, if arr = [10, 20, 20, 30] and targets = [20, 20],
            # First match: i=1, j=0, a=20, t=20. indices_out[0]=1. i++, j++.
            # Second iter: i=2, j=1, a=20, t=20. indices_out[1]=2. i++, j++. This is likely desired.
            # Let's stick to advancing both 'i' and 'j' on match for simplicity.
            i += 1

    # If we reached the end of arr but still have targets left
    while j < q:
        indices_out[j] = -1
        j += 1


@njit(parallel=True, fastmath=True, cache=True)
def parallel_vector_pivot_search(arr, targets, results, pivot_count=16):
    """
    Vector Pivot Search (VPS) with robust chunk selection.
    
    Uses binary search on pivot boundaries to find the correct chunk for each target,
    making it robust for any data distribution.
    
    Args:
        arr: Sorted numpy array to search in
        targets: Array of target values to search for
        results: Output array for results (modified in-place)
        pivot_count: Number of pivot chunks to divide array into
    """
    n = arr.size
    if n == 0: # Handle empty array
        results[:] = -1
        return
    num_targets = targets.size

    # Ensure pivot_count is reasonable
    if pivot_count < 1:
        pivot_count = 1
    pivot_count = min(pivot_count, n) # Cannot have more pivots than elements

    # Calculate chunk size based on index division
    chunk_size = (n + pivot_count - 1) // pivot_count

    # Pre-compute pivot boundaries for efficient lookup
    pivot_boundaries = np.empty(pivot_count + 1, dtype=np.int64)
    for i in range(pivot_count + 1):
        pivot_boundaries[i] = min(i * chunk_size, n)

    # Get value range for target-to-chunk mapping
    min_val = arr[0]
    max_val = arr[-1]
    val_range = max_val - min_val
    if val_range <= 0:
        val_range = 1

    # Distribution uniformity check: if data is highly non-uniform, skip chunking
    use_chunking = True
    if pivot_count > 1 and n > 100:  # Only check for larger arrays with multiple chunks
        # Sample a few points to check uniformity
        sample_indices = [n // 4, n // 2, 3 * n // 4]
        expected_values = [min_val + (val_range * idx // (n - 1)) for idx in sample_indices]
        actual_values = [arr[idx] for idx in sample_indices]
        
        # Check if actual values deviate significantly from expected uniform distribution
        max_deviation = 0
        for expected, actual in zip(expected_values, actual_values):
            deviation = abs(actual - expected) / val_range if val_range > 0 else 0
            max_deviation = max(max_deviation, deviation)
        
        # If deviation is too high, disable chunking (use full binary search)
        if max_deviation > 0.3:  # 30% deviation threshold
            use_chunking = False

    for i in prange(num_targets):
        t = targets[i]

        if use_chunking and pivot_count > 1:
            # Original research algorithm: Intelligent pivot chunk estimation
            # Estimate chunk index based on target's value relative to array's value range
            relative_pos = (float(t - min_val) / val_range) if val_range > 1 else 0.0
            relative_pos = max(0.0, min(1.0, relative_pos))  # Clamp to [0, 1]

            # Map relative value position to a chunk index
            # Fix floating-point precision issue for large arrays
            estimated_idx = int(relative_pos * (n - 1))
            estimated_idx = max(0, min(estimated_idx, n - 1))  # Clamp to valid array bounds
            chunk_idx = estimated_idx // chunk_size  # Map estimated index to chunk index

            chunk_idx = max(0, min(chunk_idx, pivot_count - 1))  # Clamp chunk index

            # Define search bounds for the chosen chunk
            left = pivot_boundaries[chunk_idx]
            right = pivot_boundaries[chunk_idx + 1] - 1
            right = min(right, n - 1)

            # Perform binary search within the selected pivot chunk
            if left <= right:
                idx = binary_search(arr, t, left, right)
                if idx != -1:
                    results[i] = idx
                else:
                    # Fallback: if chunk estimation fails, use full binary search
                    results[i] = binary_search(arr, t, 0, n - 1)
            else:
                results[i] = -1
        else:
            # For non-uniform data, use standard binary search
            results[i] = binary_search(arr, t, 0, n - 1)
