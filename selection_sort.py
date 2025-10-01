# selection_sort.py
from __future__ import annotations
from typing import Any, List

# Optional: Numba JIT (if installed) for approximate "compiler optimization" study
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

def _swap(arr: List[Any], i: int, j: int, ctr):
    if i != j:
        ctr.swaps += 1
        arr[i], arr[j] = arr[j], arr[i]

def selection_sort_instrumented(data: List[int | float], ctr) -> List[int | float]:
    """Instrumented selection sort.
    - 'ctr' must have 'comparisons' and 'swaps' integer attributes.
    """
    arr = data[:]  # work on a copy
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            ctr.comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            _swap(arr, i, min_idx, ctr)
    return arr

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def selection_sort_numba(data):
        arr = data.copy()
        n = len(arr)
        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            if min_idx != i:
                tmp = arr[i]
                arr[i] = arr[min_idx]
                arr[min_idx] = tmp
        return arr
