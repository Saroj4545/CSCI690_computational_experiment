# Quadratic Sorting Experiments  
**Comparing Insertion Sort vs Selection Sort**

## Overview  
This project is an experimental study of two quadratic complexity sorting algorithms: **insertion sort** and **selection sort**.  
The goal is to measure, analyze, and compare their performance under varying conditions such as input order, duplicates, data type, and profiling overhead.  

## Experiment Objectives  
- Compare insertion sort and selection sort in terms of:  
  - Wall-clock time  
  - CPU time  
  - Number of comparisons and swaps  
  - Operation rates (comparisons/sec, swaps/sec)  
  - Relative performance vs Python’s built-in `sorted()` (Timsort)  
- Investigate the effect of:  
  - Profiling latency  
  - Compiler optimization (Numba JIT, if available)  
  - Partially sorted data (varying adjacency correctness)  
  - Duplicate ratios (unique elements vs total elements)  
  - Integer vs floating-point data  

## Features  
- **Randomized data generation** with adjustable size, duplicates, and type.  
- **Partially sorted data** generation for testing sensitivity to initial order.  
- **Instrumented sorting algorithms** (counts comparisons and swaps).  
- **CSV output** for per-run results.  
- **Optional profiling** using `cProfile`.  
- **Statistical summaries** (range, mean, median, standard deviation).  

## Repository Contents  
- `quadratic_sorts_experiment.py` → Main experiment runner.  
- `insertion_sort.py` → Instrumented insertion sort implementation.  
- `selection_sort.py` → Instrumented selection sort implementation.  
- `results.csv` → Sample results file (generated after running).  


## How to Run  
```bash
# Run both algorithms on 2000 random integers, 5 runs each
python quadratic_sorts_experiment.py --algo both --n 2000 --runs 5 --dtype int --data random --csv results.csv

# Example: Run only insertion sort on floats with partial ordering
python quadratic_sorts_experiment.py --algo insertion --n 1000 --runs 3 --dtype float --data partial --adjacent-correct 0.75
