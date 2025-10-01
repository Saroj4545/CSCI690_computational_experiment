# insertion_sort.py
from __future__ import annotations
from typing import Any, List

# Optional: Numba JIT (if installed) for approximate "compiler optimization" study
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

def insertion_sort_instrumented(data: List[int | float], ctr) -> List[int | float]:
    """Instrumented insertion sort.
    - 'ctr' must have 'comparisons' and 'swaps' integer attributes.
    - We count a 'swap' each time we shift an element right.
    """
    arr = data[:]  # work on a copy
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0:
            ctr.comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                ctr.swaps += 1  # treat shift as a swap/move
                j -= 1
            else:
                break
        arr[j + 1] = key
    return arr

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def insertion_sort_numba(data):
        arr = data.copy()
        n = len(arr)
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
