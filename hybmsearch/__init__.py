"""
HyBMSearch: Hybrid Bayesian Mutation Search

A high-performance search algorithm optimization framework that combines:
- Multiple search strategies (binary, interpolation, chunk-based, vector pivot)
- Genetic Algorithm optimization with Bayesian mutation
- Parallel processing with Numba JIT compilation
- Comprehensive benchmarking and validation

Author: HyBMSearch Team
Version: 1.0.0
License: CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)
See: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""


__version__ = "1.0.0"
__author__ = "HyBMSearch Team"
__license__ = "CC BY-NC 4.0"

# Main API imports
from .core import perform_search, SearchConfig
from .optimization import optimize_search_parameters
from .benchmarking import benchmark_search, validate_results
from .utils import setup_logger, set_optimal_num_threads

# Algorithm imports
from .algorithms.parallel import (
    parallel_chunk_search,
    parallel_twolevel_search,
    parallel_vector_pivot_search,
    merge_search
)

from .algorithms.core import (
    binary_search,
    interpolation_search
)

from .algorithms.standard import (
    parallel_binary_search,
    parallel_interpolation_search,
    parallel_fibonacci_search
)

__all__ = [
    # Main API
    'perform_search',
    'SearchConfig',
    'optimize_search_parameters',
    'benchmark_search',
    'validate_results',
    
    # Utilities
    'setup_logger',
    'set_optimal_num_threads',
    
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
