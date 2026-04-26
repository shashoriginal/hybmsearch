<div align="center">
  <img src="HyBMSearchLogo.png" alt="HyBMSearch Logo" width="200"/>
  
  # HyBMSearch (Hybrid Bayesian Multi-Level Search)
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
  [![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.5224518-blue.svg)](https://dx.doi.org/10.2139/ssrn.5224518)
  
  *A Fast Multi-Level Search Algorithm Delivering Order-of-Magnitude Speedups on Multi-Billion Datasets*
</div>

## Authors

**Shashank Raj** - *Department of Computer Science and Engineering, Michigan State University*  
Email: [rajshash@msu.edu](mailto:rajshash@msu.edu)

**Kalyanmoy Deb** - *Department of Electrical and Computer Engineering, Michigan State University*  
Email: [kdeb@msu.edu](mailto:kdeb@msu.edu)

---

HyBMSearch is a high-performance search algorithm optimization framework that combines multiple search strategies with genetic algorithm optimization and Bayesian mutation for automatic parameter tuning. This research-grade implementation delivers dramatic speedups on arrays ranging from 100 million to 10 billion elements, achieving up to 700× performance improvements over NumPy's optimized `searchsorted` function.

## Key Features

- **Multiple Search Algorithms**: Binary, interpolation, chunk-based, vector pivot search, and merge search
- **Automatic Parameter Optimization**: Genetic Algorithm with Bayesian optimization for intelligent parameter tuning  
- **High Performance**: Numba JIT compilation with parallel processing support
- **Comprehensive Benchmarking**: Built-in comparison against NumPy and standard parallel algorithms
- **Easy to Use**: Clean API with sensible defaults and extensive configuration options
- **Research-Grade**: Peer-reviewed implementation with published results
- **Scalable**: Tested on datasets up to 10 billion elements
- **Robust**: 100% correctness across all data distributions

## Installation
```bash
# Extract the supplementary code package
# Navigate to the HyBMSearch directory
cd HyBMSearch
pip install -e .
```

**Note**: This is an interim research code release. Please respect the CC-BY-NC license terms.

## Experiment Scripts

This interim release includes two key experiment scripts that replicate the paper's results:

### 1. `paper_experiments.py`
**Purpose**: Replicates the main sequential benchmark experiments from the paper.

**What it does**:
- Tests array sizes: 5B and 10B elements (configurable)
- Uses sequential (uniform) data distribution
- Runs GA optimization with Bayesian mutation to find optimal parameters
- Benchmarks HyBMSearch against NumPy, Parallel Binary, Parallel Interpolation, and Parallel Fibonacci
- Validates correctness against NumPy baseline
- Saves results in structured JSON format for analysis

**Usage**:
```bash
python paper_experiments.py
```

**Output**: Creates `paper_results_YYYYMMDD_HHMMSS/` directory with:
- Optimized parameters for each array size
- Benchmark timing results
- GA optimization history
- Validation results

### 2. `benchmark_1b_int32.py`
**Purpose**: Tests algorithm robustness across different data distributions.

**What it does**:
- Tests on 1 billion elements using int32 for memory efficiency (3.73 GB per array)
- Tests 5 different data distributions:
  - **Sequential**: Perfect uniform data (0, 1, 2, ...)
  - **Random Uniform**: Random uniform distribution
  - **Sparse**: Data with large gaps
  - **Exponential**: Exponential distribution
  - **Bimodal**: Two-peak distribution
- Runs GA optimization for each distribution
- Validates algorithm performance across diverse data patterns

**Usage**:
```bash
python benchmark_1b_int32.py
```

**Output**: Creates `benchmark_1b_int32_YYYYMMDD_HHMMSS/` directory with:
- Results for each distribution type
- Performance comparisons and speedup calculations
- Memory usage statistics
- Robustness validation results

## Quick Start

```python
import numpy as np
from hybmsearch import perform_search, SearchConfig

# Create test data
arr = np.arange(1000000)  # Sorted array
targets = np.random.randint(0, 1000000, 10000)  # Target values

# Basic usage with default configuration
results, elapsed_time = perform_search(arr, targets)
print(f"Search completed in {elapsed_time:.4f} seconds")

# Using custom configuration
config = SearchConfig(
    use_vector_pivot=True,
    pivot_count=32,
    num_threads=4
)
results, elapsed_time = perform_search(arr, targets, config=config)
```

## Algorithm Overview

### Core Search Strategies

1. **Chunk-based Search**: Divides array into chunks and estimates target locations
2. **Two-level Hierarchical Search**: Hierarchical chunking for very large arrays
3. **Vector Pivot Search (VPS)**: Pivot-based chunk selection with parallel processing
4. **Merge Search**: Linear-time search for pre-sorted targets
5. **Standard Algorithms**: Binary, interpolation, and Fibonacci search for comparison

### Optimization Framework

- **Genetic Algorithm**: Population-based optimization with tournament selection
- **Bayesian Mutation**: Uses Gaussian Process Regression with Expected Improvement acquisition function
- **Parameter Space**: Optimizes chunk sizes, thread counts, algorithm selection, and more
- **Adaptive Learning**: Learns from previous evaluations to suggest better parameters

## Configuration Options

The `SearchConfig` class provides extensive configuration:

```python
config = SearchConfig(
    use_merge_search=False,      # Use merge search for sorted targets
    num_levels=1,                # Number of chunk levels (1 or 2)
    chunk_size=1000000,          # Primary chunk size
    sub_chunk_size=100000,       # Sub-chunk size for 2-level search
    use_interpolation=False,     # Use interpolation within chunks
    num_threads=1,               # Number of parallel threads
    use_vector_pivot=False,      # Enable Vector Pivot Search
    pivot_count=16               # Number of pivots for VPS
)
```

## Benchmarking

```python
from hybmsearch.benchmarking import run_comprehensive_benchmark
from hybmsearch.utils import create_test_data, setup_logger

# Setup logging
logger = setup_logger("benchmark_results.txt")

# Create test data
arr, targets = create_test_data(size=1000000, data_type="random")

# Run comprehensive benchmark
results = run_comprehensive_benchmark(arr, targets, config=config)

# Results include timing for HyBMSearch, NumPy, and standard parallel algorithms
for method, data in results.items():
    if 'time' in data:
        print(f"{method}: {data['time']:.6f}s")
```

## API Reference

### Core Functions

- `perform_search(arr, targets, config=None, **kwargs)`: Main search interface
- `optimize_search_parameters(arr, targets, pop_size=50, ngen=10)`: GA optimization

### Configuration

- `SearchConfig`: Configuration dataclass for all search parameters

### Benchmarking

- `benchmark_search()`: Benchmark HyBMSearch with configuration
- `benchmark_numpy_search()`: Benchmark NumPy searchsorted baseline
- `validate_results()`: Compare results for correctness
- `run_comprehensive_benchmark()`: Full benchmark suite

### Utilities

- `setup_logger()`: Configure logging for benchmarks
- `set_optimal_num_threads()`: Set Numba thread count
- `create_test_data()`: Generate test arrays and targets

## Performance Considerations

- **Data Distribution**: Algorithms work best on uniformly distributed data
- **Array Size**: Optimized for large arrays (millions to billions of elements)
- **Memory Usage**: Efficient memory usage with in-place result arrays
- **Thread Scaling**: Performance scales with available CPU cores

## Requirements

- Python 3.8+
- NumPy >= 1.19.0
- Numba >= 0.56.0
- scikit-learn >= 1.0.0
- SciPy >= 1.7.0
- DEAP >= 1.3.0

## Important Notice: Final Journal Code

**⚠️ This is the final code release used in the journal publication.**

Minor adjustments may have been made during packaging to improve code
readability.

This code is provided for research and academic purposes only. It is protected by the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

**License Terms (MUST BE RESPECTED):**
- **Attribution Required**: You must give appropriate credit to the authors
- **Non-Commercial Use Only**: You may NOT use this work for commercial purposes
- **Derivative Works Allowed**: You may create derivative works with proper attribution
- **No Commercialization**: You may NOT commercialize either the original work or any derivative work

**For Research Use:** If you use HyBMSearch in academic research, please cite our paper and include proper attribution. For commercial licensing, please contact the authors directly.

See [LICENSE](LICENSE) for the full legal text.

## Contributing

**Note**: This is an interim research code release. Please respect the CC-BY-NC license terms.

### For Researchers
- **Academic Use**: Feel free to use HyBMSearch for research purposes with proper citation
- **Bug Reports**: Please contact the authors directly for any issues
- **Research Collaboration**: Contact the authors for research collaborations

### For Developers
- **Code Contributions**: Please contact the authors before making any contributions
- **Commercial Use**: Requires explicit permission from the authors
- **Derivative Works**: Permitted for non-commercial use with proper attribution

### Contact
- **Shashank Raj**: [rajshash@msu.edu](mailto:rajshash@msu.edu)
- **Kalyanmoy Deb**: [kdeb@msu.edu](mailto:kdeb@msu.edu)

## Citation

If you use this work, please cite:

```bibtex
@article{raj2026hybmsearch,
  title={HyBMSearch: A Fast Multi-Level Search Algorithm Delivering Order-of-Magnitude Speedups on Multi-Billion Datasets},
  author={Raj, Shashank and Deb, Kalyanmoy},
  journal={Journal of Parallel and Distributed Computing},
  pages={105226},
  year={2026},
  publisher={Elsevier}
}
```

## Research Paper

This interim code release accompanies our research paper in the *Journal of Parallel and Distributed Computing* (2026).

> **"HyBMSearch: A Fast Multi-Level Search Algorithm Delivering Order-of-Magnitude Speedups on Multi-Billion Datasets"**  
> *Shashank Raj, Kalyanmoy Deb*  
> Journal of Parallel and Distributed Computing, 2026

**Key Results:**
- **Up to 700× speedup** over NumPy's `searchsorted`
- **Tested on 10 billion elements** with 1 billion queries
- **Automatic parameter optimization** via GA + Bayesian methods
- **100% correctness** across all data distributions

## Acknowledgments

We acknowledge and thank the resources provided by Computational Optimization and Innovation Laboratory.

### Computational Resources
Benchmarking was performed on high-performance computing clusters with:
- 2× AMD EPYC 7H12 @ 2.6 GHz (256 virtual cores)
- 1024 GB RAM
- AlmaLinux 8 operating system

## Support

For questions, issues, or collaboration requests:
- **Technical Issues**: Please contact the authors directly
- **Research Inquiries**: [rajshash@msu.edu](mailto:rajshash@msu.edu)
- **Commercial Licensing**: [kdeb@msu.edu](mailto:kdeb@msu.edu)

---

<div align="center">
  <p><strong>HyBMSearch</strong> - Interim Research Code Release</p>
  <p>© 2026 Shashank Raj & Kalyanmoy Deb. All rights reserved.</p>
  <p><em>Protected by CC-BY-NC 4.0 License</em></p>
</div>
