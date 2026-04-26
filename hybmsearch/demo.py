"""
HyBMSearch Demo Script

Demonstrates the main functionality of the HyBMSearch package.
"""

import numpy as np
import time
from .core import perform_search, SearchConfig
from .optimization import optimize_search_parameters
from .benchmarking import run_comprehensive_benchmark
from .utils import setup_logger, create_test_data, get_system_info


def demo_basic_search():
    """Demonstrate basic search functionality."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Search Functionality")
    print("="*60)
    
    # Create test data
    size = 100000
    arr, targets = create_test_data(size=size, target_count=1000, data_type="sequential")
    
    print(f"Created array of size {size:,} and {len(targets):,} targets")
    
    # Basic search with default configuration
    print("\n--- Default Configuration ---")
    start = time.time()
    results, elapsed = perform_search(arr, targets)
    setup_time = time.time() - start
    print(f"Search time: {elapsed:.6f}s (total: {setup_time:.6f}s)")
    print(f"Found {np.sum(results != -1)} out of {len(targets)} targets")
    
    # Search with Vector Pivot configuration
    print("\n--- Vector Pivot Search ---")
    config = SearchConfig(use_vector_pivot=True, pivot_count=16)
    results, elapsed = perform_search(arr, targets, config=config)
    print(f"Search time: {elapsed:.6f}s")
    print(f"Found {np.sum(results != -1)} out of {len(targets)} targets")
    
    # Search with Merge Search (requires sorted targets)
    print("\n--- Merge Search ---")
    config = SearchConfig(use_merge_search=True)
    results, elapsed = perform_search(arr, targets, config=config)
    print(f"Search time: {elapsed:.6f}s")
    print(f"Found {np.sum(results != -1)} out of {len(targets)} targets")


def demo_optimization():
    """Demonstrate genetic algorithm optimization."""
    print("\n" + "="*60)
    print("DEMO 2: Genetic Algorithm Optimization")
    print("="*60)
    
    # Create test data
    size = 500000
    arr, targets = create_test_data(size=size, target_count=5000, data_type="random")
    
    print(f"Created array of size {size:,} and {len(targets):,} targets")
    print("Running GA optimization (this may take a few minutes)...")
    
    # Run optimization with small population for demo
    best_params, cache = optimize_search_parameters(
        arr, targets, 
        pop_size=10,  # Small for demo
        ngen=3        # Few generations for demo
    )
    
    print(f"\nBest parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Test the optimized configuration
    print("\n--- Testing Optimized Configuration ---")
    config = SearchConfig(**best_params)
    results, elapsed = perform_search(arr, targets, config=config)
    print(f"Optimized search time: {elapsed:.6f}s")
    print(f"Found {np.sum(results != -1)} out of {len(targets)} targets")


def demo_comprehensive_benchmark():
    """Demonstrate comprehensive benchmarking."""
    print("\n" + "="*60)
    print("DEMO 3: Comprehensive Benchmarking")
    print("="*60)
    
    # Create test data
    size = 200000  # Smaller for demo
    arr, targets = create_test_data(size=size, target_count=2000, data_type="sparse")
    
    print(f"Created sparse array of size {size:,} and {len(targets):,} targets")
    
    # Test different configurations
    configs = [
        ("Default", SearchConfig()),
        ("Vector Pivot", SearchConfig(use_vector_pivot=True, pivot_count=8)),
        ("Two-Level", SearchConfig(num_levels=2, chunk_size=50000, sub_chunk_size=5000)),
        ("Interpolation", SearchConfig(use_interpolation=True, chunk_size=25000))
    ]
    
    for name, config in configs:
        print(f"\n--- {name} Configuration ---")
        results = run_comprehensive_benchmark(arr, targets, config=config)
        
        # Print timing summary
        print(f"\nTiming Results for {name}:")
        for method, data in results.items():
            if 'time' in data:
                print(f"  {method:20s}: {data['time']:.6f}s")


def demo_data_types():
    """Demonstrate performance on different data distributions."""
    print("\n" + "="*60)
    print("DEMO 4: Performance on Different Data Types")
    print("="*60)
    
    size = 100000
    target_count = 1000
    
    data_types = ["sequential", "random", "sparse"]
    config = SearchConfig(use_vector_pivot=True, pivot_count=16)
    
    for data_type in data_types:
        print(f"\n--- {data_type.title()} Data ---")
        arr, targets = create_test_data(size=size, target_count=target_count, data_type=data_type)
        
        # HyBMSearch
        results, elapsed = perform_search(arr, targets, config=config)
        found = np.sum(results != -1)
        print(f"HyBMSearch: {elapsed:.6f}s, found {found}/{target_count}")
        
        # NumPy baseline for comparison
        start = time.time()
        indices = np.searchsorted(arr, targets, side='left')
        valid_mask = (indices < len(arr)) & (arr[indices] == targets)
        np_results = np.where(valid_mask, indices, -1)
        np_elapsed = time.time() - start
        np_found = np.sum(np_results != -1)
        print(f"NumPy:      {np_elapsed:.6f}s, found {np_found}/{target_count}")
        
        speedup = np_elapsed / elapsed if elapsed > 0 else float('inf')
        print(f"Speedup:    {speedup:.2f}x")


def main():
    """Run all demos."""
    # Setup logging
    logger = setup_logger("hybmsearch_demo.txt")
    
    print("HyBMSearch Demo Script")
    print("=====================")
    
    # Print system info
    info = get_system_info()
    print(f"System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    try:
        # Run demos
        demo_basic_search()
        demo_optimization()
        demo_comprehensive_benchmark()
        demo_data_types()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("Check 'hybmsearch_demo.txt' for detailed logs.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
