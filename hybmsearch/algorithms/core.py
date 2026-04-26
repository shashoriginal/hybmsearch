"""
Core Search Algorithms

Basic search algorithm implementations using Numba JIT compilation.
These are the fundamental building blocks used by parallel algorithms.
"""

import numpy as np
from numba import njit


@njit
def binary_search(arr, target, left, right):
    """
    Plain binary search implementation.
    
    Args:
        arr: Sorted numpy array to search in
        target: Value to search for
        left: Left boundary index (inclusive)
        right: Right boundary index (inclusive)
        
    Returns:
        int: Index of target if found, -1 if not found
    """
    while left <= right:
        mid = (left + right) >> 1
        # Check array bounds before accessing arr[mid]
        if mid < 0 or mid >= arr.size:
             # This case should ideally not happen with correct left/right logic,
             # but added as a safeguard.
             return -1
        val = arr[mid]
        if val == target:
            return mid
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


@njit
def interpolation_search(arr, target, left, right):
    """
    Interpolation search for near-uniform data.
    
    Works best when data is uniformly distributed. Falls back gracefully
    when interpolation estimates go out of bounds.
    
    Args:
        arr: Sorted numpy array to search in
        target: Value to search for
        left: Left boundary index (inclusive)
        right: Right boundary index (inclusive)
        
    Returns:
        int: Index of target if found, -1 if not found
    """
    # Ensure indices are valid before starting loop
    if left < 0 or right >= arr.size or left > right:
        return -1

    while left <= right and target >= arr[left] and target <= arr[right]:
        # Check if range has collapsed or values are identical
        if arr[right] == arr[left]:
            if arr[left] == target:
                return left
            else:
                return -1 # Target cannot be in this range

        # Calculate position using interpolation formula
        # Ensure denominator is not zero (handled by arr[right] == arr[left] check above)
        # Use floating point division for calculation
        pos = left + int( (float(right - left) / (arr[right] - arr[left])) * (target - arr[left]) )

        # Crucial bound check for the calculated position
        if pos < left or pos > right:
             # Fallback to binary search if pos is out of bounds for safety
             # Or return -1 if strict interpolation is required and this indicates an issue
             # Let's return -1 to signal interpolation estimate failed boundary check
             # Revert to Binary Search might be safer in practice if data isn't perfectly uniform
             # For now, stick to pure interpolation intent:
             # return binary_search(arr, target, left, right) # Fallback option
             return -1 # Stricter: interpolation calculation failed boundary

        val = arr[pos]
        if val == target:
            return pos
        elif val < target:
            left = pos + 1
        else:
            right = pos - 1

    # After loop, check if left points to target (can happen if loop terminates early)
    if left <= right and left < arr.size and arr[left] == target:
        return left

    return -1
