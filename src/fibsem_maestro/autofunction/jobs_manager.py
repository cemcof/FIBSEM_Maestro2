# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.autofunction.result import AutofunctionResult


class JobsManager:
    """
    Manages concurrent resolution calculation jobs during an autofunction sweep.

    Submits resolution calculations to a thread pool as images are acquired
    and collects their results. Results are read from the futures themselves,
    never from done-callbacks: a callback may run after `Future.result()` has
    already returned in another thread, so a result stored by a callback can
    be missing when the jobs are reported as finished.

    Args:
        executor: Optional external executor. If omitted, an internal
            ThreadPoolExecutor is created and owned by this instance.
    """

    def __init__(
        self,
        executor: Executor | None = None,
    ) -> None:
        self._own_executor = executor is None
        self._executor: Executor = executor or ThreadPoolExecutor()

        self._lock = threading.Lock()
        self._pending: list[Future[AutofunctionResult]] = []
        self._results: list[AutofunctionResult] = []

    def submit(self, fn: Callable[[], AutofunctionResult]) -> None:
        """
        Submit a callable for asynchronous resolution calculation.

        Exceptions raised by the callable are discarded when collecting: a
        failed job contributes no result.

        Args:
            fn: A zero-argument callable returning an AutofunctionResult.
        """
        future = self._executor.submit(fn)
        with self._lock:
            self._pending.append(future)

    def wait_and_collect(self) -> list[AutofunctionResult]:
        """
        Block until all submitted jobs finish and return the results.

        Returns:
            All successfully computed AutofunctionResult instances. Failed jobs
            are excluded.
        """
        self._collect(wait=True)
        with self._lock:
            return list(self._results)

    def collect_completed(self) -> list[AutofunctionResult]:
        """
        Return results of already-finished jobs without waiting or clearing.

        Returns:
            All AutofunctionResult instances computed so far.
        """
        self._collect(wait=False)
        with self._lock:
            return list(self._results)

    def wait(self) -> None:
        """
        Block until all submitted jobs finish, but do not return their results.
        """
        self._collect(wait=True)

    def wait_and_clear(self) -> None:
        """
        Block until all pending jobs finish, then discard all results.
        """
        self._collect(wait=True)
        with self._lock:
            self._results = []

    def _collect(self, wait: bool) -> None:
        """
        Move the results of finished jobs from the pending futures to the results.

        Args:
            wait: Wait for every pending job, including those submitted while
                waiting; otherwise take only the jobs finished by now.
        """
        while True:
            with self._lock:
                if wait:
                    batch, self._pending = self._pending, []
                else:
                    finished = [(f, f.done()) for f in self._pending]
                    batch = [f for f, done in finished if done]
                    self._pending = [f for f, done in finished if not done]

            if not batch:
                return

            for future in batch:
                try:
                    result = future.result()
                except Exception:
                    continue  # the job logged its own failure
                with self._lock:
                    self._results.append(result)

            if not wait:
                return

    def __enter__(self) -> Self:
        """Enter the context manager, returning this instance."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, shutting down the executor if owned."""
        if self._own_executor:
            self._executor.shutdown(wait=True)
