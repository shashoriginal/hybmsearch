"""
HyBMSearch Algorithms Package

Contains all search algorithm implementations:
- core: Basic search algorithms (binary, interpolation)
- parallel: Numba-optimized parallel algorithms 
- standard: ThreadPool-based standard algorithm implementations
"""

from .core import binary_search, interpolation_search
from .parallel import (
    parallel_chunk_search,
    parallel_twolevel_search,
    parallel_vector_pivot_search,
    merge_search
)
from .standard import (
    parallel_binary_search,
    parallel_interpolation_search,
    parallel_fibonacci_search
)

__all__ = [
    # Core algorithms
    'binary_search',
    'interpolation_search',
    
    # Parallel algorithms
    'parallel_chunk_search',
    'parallel_twolevel_search',
    'parallel_vector_pivot_search',
    'merge_search',
    
    # Standard algorithms
    'parallel_binary_search',
    'parallel_interpolation_search',
    'parallel_fibonacci_search',
]
