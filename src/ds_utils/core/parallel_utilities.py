import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar, TypedDict


# region Aux Class -------------------------------------------------------------

T = TypeVar("T")
R = TypeVar("R")

class ParallelResult[T, R](TypedDict):
    item: T
    result: R


class ParallelError[T](TypedDict):
    item: T
    error: dict[str, str]

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

CPU_THREADS = os.cpu_count() or 1

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Aux Functions ---------------------------------------------------------


# region Thread Functions ------------------------------------------------------

def parallelize(
    func: Callable[[T], R],
    iterable: Iterable[T],
    max_workers: int = CPU_THREADS,
) -> tuple[list[ParallelResult], list[ParallelError]]:
    """
    Execute a function concurrently for each item in an iterable.

    Parameters
    ----------
    func : Callable[[T], R]
        Function to execute for each item. Receives an input of type T
        and returns a result of type R.
    iterable : Iterable[T]
        Iterable containing the input items to process.
    max_workers : int, default=CPU_THREADS
        Maximum number of concurrent threads.

    Returns
    -------
    results : list[ParallelResult]
        Successful executions. Each dictionary contains the original
        input item under 'item' and the function output under 'result'.
    errors : list[ParallelError]
        Failed executions. Each dictionary contains the original input
        item under 'item' and the exception details under 'error'.

    Notes
    -----
    Results and errors are collected in completion order, not input order.
    Exceptions raised by the executed function are captured and stored
    in the errors list.
    """
    results: list[ParallelResult] = []
    errors: list[ParallelError] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(func, item): item
            for item in iterable
        }

        for future in as_completed(futures):
            item = futures[future]

            try:
                results.append({
                    "item": item,
                    "result": future.result(),
                })

            except Exception as exc:
                errors.append({
                    "item": item,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                })

    return results, errors

# endregion Thread Functions ---------------------------------------------------


# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------