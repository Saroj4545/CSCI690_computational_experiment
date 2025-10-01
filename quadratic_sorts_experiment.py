
"""
Quadratic Sorting Experiment Suite (modular)
- Imports instrumented sorts from insertion_sort.py and selection_sort.py
"""

from __future__ import annotations
import argparse
import csv
import random
import statistics as stats
import time
import cProfile
import pstats
import io
from dataclasses import dataclass, asdict
from typing import Callable, List, Tuple, Dict, Any

# Import the two sorts from separate files
from insertion_sort import insertion_sort_instrumented
from selection_sort import selection_sort_instrumented

# ---------------------------- Utilities & Data Gen ---------------------------- #

def generate_random_data(n: int, *, dtype: str = "int", seed: int | None = None,
                         unique_ratio: float = 1.0) -> List[float | int]:
    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random

    unique_count = max(1, min(n, int(round(unique_ratio * n))))
    pool = [rnd.randint(-10*n, 10*n) for _ in range(unique_count)]
    data = [pool[rnd.randrange(unique_count)] for _ in range(n)]

    if dtype == "float":
        data = [float(x) for x in data]
    elif dtype != "int":
        raise ValueError("dtype must be 'int' or 'float'")

    return data


def make_partially_sorted(data: List[int | float], adjacent_correct: float) -> List[int | float]:
    n = len(data)
    if n < 2:
        return data[:]
    target = max(0.0, min(1.0, adjacent_correct))
    base = sorted(data)
    k = int(round((1.0 - target) * (n - 1)))
    rnd = random.Random()
    arr = base[:]
    for _ in range(k):
        i = rnd.randrange(n - 1)
        arr[i], arr[i+1] = arr[i+1], arr[i]
    return arr


def adjacency_correct_fraction(arr: List[int | float]) -> float:
    if len(arr) < 2:
        return 1.0
    ok = 0
    for i in range(len(arr) - 1):
        if arr[i] <= arr[i+1]:
            ok += 1
    return ok / (len(arr) - 1)

# ---------------------------- Instrumentation ---------------------------- #

@dataclass
class Counters:
    comparisons: int = 0
    swaps: int = 0
    def reset(self):
        self.comparisons = 0
        self.swaps = 0

# ---------------------------- Measurement Harness ---------------------------- #

@dataclass
class RunResult:
    algo: str
    n: int
    dtype: str
    unique_ratio: float
    adjacent_correct: float
    wall_time_s: float
    cpu_time_s: float
    comparisons: int
    swaps: int
    cmp_rate_per_s: float
    swap_rate_per_s: float
    builtin_wall_s: float
    speed_vs_builtin: float
    correct: bool
    seed_used: int
    adjacency_fraction_start: float
    profile_seconds: float | None = None


def measure_once(
    algo_name: str,
    data: List[int | float],
    dtype: str,
    unique_ratio: float,
    adjacent_correct_target: float,
    profile: bool,
    jit: str,
) -> RunResult:
    ctr = Counters()

    if algo_name == "insertion":
        algo_fn: Callable[[List[Any], Counters], List[Any]] = insertion_sort_instrumented
    elif algo_name == "selection":
        algo_fn = selection_sort_instrumented
    else:
        raise ValueError("algo_name must be 'insertion' or 'selection'")

    # Builtin baseline (Timsort)
    data_for_builtin = data[:]
    t0b = time.perf_counter()
    builtin_sorted = sorted(data_for_builtin)
    t1b = time.perf_counter()
    builtin_wall = t1b - t0b

    prof = None

    adj_frac_start = adjacency_correct_fraction(data)

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    if profile:
        prof = cProfile.Profile()
        prof.enable()

    sorted_out = algo_fn(data, ctr)

    if profile:
        prof.disable()

    t1_wall = time.perf_counter()
    t1_cpu = time.process_time()

    wall = t1_wall - t0_wall
    cpu = t1_cpu - t0_cpu

    cmp_rate = ctr.comparisons / wall if wall > 0 else float('inf')
    swap_rate = ctr.swaps / wall if wall > 0 else float('inf')
    correct = (sorted_out == builtin_sorted)

    prof_seconds = None
    if profile and prof is not None:
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s)
        ps.strip_dirs().sort_stats("cumulative").print_stats(20)
        prof_text = s.getvalue()
        prof_seconds = wall
        stamp = int(time.time() * 1000)
        with open(f"profile_{algo_name}_{stamp}.txt", "w", encoding="utf-8") as f:
            f.write(prof_text)

    speed_vs_builtin = wall / builtin_wall if builtin_wall > 0 else float('inf')

    return RunResult(
        algo=algo_name,
        n=len(data),
        dtype=dtype,
        unique_ratio=unique_ratio,
        adjacent_correct=adjacent_correct_target,
        wall_time_s=wall,
        cpu_time_s=cpu,
        comparisons=ctr.comparisons,
        swaps=ctr.swaps,
        cmp_rate_per_s=cmp_rate,
        swap_rate_per_s=swap_rate,
        builtin_wall_s=builtin_wall,
        speed_vs_builtin=speed_vs_builtin,
        correct=correct,
        seed_used=0,  # filled by caller
        adjacency_fraction_start=adj_frac_start,
        profile_seconds=prof_seconds,
    )

# ---------------------------- Experiment Runner ---------------------------- #

def run_experiment(
    algo_list: List[str],
    n: int,
    runs: int,
    dtype: str,
    unique_ratio: float,
    data_mode: str,
    adjacent_correct: float,
    repeat_same_data: bool,
    seed: int | None,
    profile: str,
    csv_path: str,
    jit: str,
):
    assert data_mode in {"random", "partial"}
    assert profile in {"off", "on"}

    base_seed = seed if seed is not None else random.randrange(1, 2**31-1)
    print(f"Base seed: {base_seed}")

    base_data = None
    if repeat_same_data:
        base_data = generate_random_data(n, dtype=dtype, seed=base_seed, unique_ratio=unique_ratio)
        if data_mode == "partial":
            base_data = make_partially_sorted(base_data, adjacent_correct)

    results: List[RunResult] = []

    fieldnames = list(RunResult.__annotations__.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for r in range(runs):
            seed_r = base_seed if repeat_same_data else base_seed + r + 1

            if base_data is not None:
                data = base_data[:]
            else:
                data = generate_random_data(n, dtype=dtype, seed=seed_r, unique_ratio=unique_ratio)
                if data_mode == "partial":
                    data = make_partially_sorted(data, adjacent_correct)

            # If you want to warm up any JIT here, you can import the numba variants
            # from the modules and call them on a tiny array if available:
            if jit in {"on", "auto"}:
                try:
                    from insertion_sort import insertion_sort_numba  # type: ignore
                    from selection_sort import selection_sort_numba  # type: ignore
                    # warm-up compiles them
                    _ = insertion_sort_numba([1, 0, 2])
                    _ = selection_sort_numba([1, 0, 2])
                except Exception:
                    if jit == "on":
                        print("[WARN] --jit on requested but numba not available; continuing without JIT.")

            for algo in algo_list:
                res = measure_once(
                    algo_name=algo,
                    data=data,
                    dtype=dtype,
                    unique_ratio=unique_ratio,
                    adjacent_correct_target=adjacent_correct,
                    profile=(profile == "on"),
                    jit=jit,
                )
                res.seed_used = seed_r
                results.append(res)
                writer.writerow(asdict(res))
                print(f"Run {r+1:02d}/{runs} | {algo:9s} | n={n} | wall={res.wall_time_s:.6f}s | "
                      f"cpu={res.cpu_time_s:.6f}s | cmp={res.comparisons} | swaps={res.swaps} | "
                      f"vs_builtin={res.speed_vs_builtin:.2f}x | correct={res.correct}")

    # Post-run statistics summary
    for algo in algo_list:
        subset = [x for x in results if x.algo == algo]
        if not subset:
            continue
        print("\n" + "="*72)
        print(f"Algorithm: {algo} | n={n} | runs={len(subset)} | dtype={dtype} | unique_ratio={unique_ratio} "
              f"| data_mode={data_mode} | adj_correct_target={adjacent_correct}")

        def S(getter, label):
            values = [getter(x) for x in subset]
            vmin, vmax = min(values), max(values)
            mean = stats.mean(values)
            median = stats.median(values)
            stdev = stats.pstdev(values) if len(values) > 1 else 0.0
            print(f"{label:22s} range=({vmin:.6f}, {vmax:.6f}) | mean={mean:.6f} | "
                  f"median={median:.6f} | std={stdev:.6f}")

        S(lambda x: x.wall_time_s,        "Wall time (s)")
        S(lambda x: x.cpu_time_s,         "CPU time (s)")
        S(lambda x: x.comparisons,        "Comparisons (count)")
        S(lambda x: x.swaps,              "Swaps (count)")
        S(lambda x: x.cmp_rate_per_s,     "Cmp rate (/s)")
        S(lambda x: x.swap_rate_per_s,    "Swap rate (/s)")
        S(lambda x: x.builtin_wall_s,     "Builtin wall (s)")
        S(lambda x: x.speed_vs_builtin,   "Rel to builtin (x)")
        S(lambda x: x.adjacency_fraction_start, "Adjacency correct start")
        if any(x.profile_seconds is not None for x in subset):
            S(lambda x: (x.profile_seconds or 0.0), "Profile wall (s)")

    print(f"\nPer-run metrics saved to: {csv_path}")
    print("Profile summaries (if profiling enabled) saved as profile_*.txt")

# ---------------------------- CLI ---------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Quadratic Sorting Experiment Suite")
    p.add_argument("--algo", choices=["insertion", "selection", "both"], default="both",
                   help="Which algorithm(s) to run")
    p.add_argument("--n", type=int, default=2000, help="Number of elements")
    p.add_argument("--runs", type=int, default=5, help="Number of runs per algorithm")
    p.add_argument("--dtype", choices=["int", "float"], default="int",
                   help="Data element type")
    p.add_argument("--data", choices=["random", "partial"], default="random",
                   help="Random data or partially sorted data")
    p.add_argument("--adjacent-correct", type=float, default=0.5,
                   help="Target fraction of adjacent pairs initially ordered (for --data partial)")
    p.add_argument("--unique-ratio", type=float, default=1.0,
                   help="Approximate ratio of unique elements (duplicates effect)")
    p.add_argument("--repeat-same-data", action="store_true",
                   help="Repeat sorts on identical data each run (same seed)")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed (optional)")
    p.add_argument("--profile", choices=["off", "on"], default="off",
                   help="Enable cProfile to observe profiling latency/overhead")
    p.add_argument("--csv", default="results.csv", help="CSV output path")
    p.add_argument("--jit", choices=["off", "on", "auto"], default="auto",
                   help="Use Numba JIT if available (auto: use if installed)")
    return p.parse_args()

def main():
    args = parse_args()
    algo_list = ["insertion", "selection"] if args.algo == "both" else [args.algo]

    run_experiment(
        algo_list=algo_list,
        n=args.n,
        runs=args.runs,
        dtype=args.dtype,
        unique_ratio=max(1e-6, min(1.0, args.unique_ratio)),
        data_mode=args.data,
        adjacent_correct=args.adjacent_correct,
        repeat_same_data=bool(args.repeat_same_data),
        seed=args.seed,
        profile=args.profile,
        csv_path=args.csv,
        jit=args.jit,
    )

if __name__ == "__main__":
    main()
