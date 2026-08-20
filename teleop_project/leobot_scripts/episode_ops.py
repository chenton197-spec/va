"""Nonblocking episode finalization for interactive collection applications."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Callable

from .recorder import EpisodeRecorder


class EpisodeOperation(str, Enum):
    """One long-running operation owned by an episode coordinator."""

    FINISH = "finish"
    DISCARD = "discard"


@dataclass(frozen=True)
class EpisodeOperationResult:
    """Completed asynchronous operation and its optional episode index."""

    operation: EpisodeOperation
    episode_index: int | None = None
    error: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class AsyncEpisodeFinalizer:
    """Run episode commit/discard away from an interactive command loop.

    The recorder itself remains the serialization authority. This helper only
    ensures that an application has at most one long-running finish or discard
    request at a time and can poll its completion without blocking Ctrl+C.
    """

    def __init__(self, recorder: EpisodeRecorder) -> None:
        self._recorder = recorder
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="leobot-finalize")
        self._lock = threading.Lock()
        self._operation: EpisodeOperation | None = None
        self._future: Future[int | None] | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._future is not None

    @property
    def operation(self) -> EpisodeOperation | None:
        with self._lock:
            return self._operation

    def finish(self) -> None:
        self._submit(EpisodeOperation.FINISH, self._recorder.stop_episode)

    def discard(self) -> None:
        def run_discard() -> int | None:
            self._recorder.discard_episode()
            return None

        self._submit(EpisodeOperation.DISCARD, run_discard)

    def poll(self) -> EpisodeOperationResult | None:
        """Return a completed operation once, or ``None`` while it is running."""

        with self._lock:
            future = self._future
            operation = self._operation
            if future is None or operation is None or not future.done():
                return None
            self._future = None
            self._operation = None
        try:
            return EpisodeOperationResult(operation=operation, episode_index=future.result())
        except BaseException as exc:
            return EpisodeOperationResult(operation=operation, error=exc)

    def wait(self, timeout_s: float | None = None) -> EpisodeOperationResult | None:
        """Wait for an active operation during controlled application shutdown."""

        with self._lock:
            future = self._future
            operation = self._operation
        if future is None or operation is None:
            return None
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return None
        except BaseException as exc:
            result_info = EpisodeOperationResult(operation=operation, error=exc)
        else:
            result_info = EpisodeOperationResult(operation=operation, episode_index=result)
        with self._lock:
            if self._future is future:
                self._future = None
                self._operation = None
        return result_info

    def close(self) -> None:
        """Release the worker after callers have stopped the robot safely."""

        self._executor.shutdown(wait=True, cancel_futures=False)

    def _submit(self, operation: EpisodeOperation, callback: Callable[[], int | None]) -> None:
        with self._lock:
            if self._future is not None:
                raise RuntimeError(f"Episode {self._operation.value} is already in progress")
            if not self._recorder.active:
                raise RuntimeError("No active episode")
            self._operation = operation
            self._future = self._executor.submit(callback)
