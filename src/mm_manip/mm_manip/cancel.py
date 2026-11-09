import time
from collections.abc import Callable

from .errors import Canceled

POLL = 0.02  # s


class CancelToken:
    def __init__(self, is_canceled: Callable[[], bool] = lambda: False):
        self._is_canceled = is_canceled

    @property
    def canceled(self) -> bool:
        return bool(self._is_canceled())

    def check(self) -> None:
        if self.canceled:
            raise Canceled()

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(min(POLL, max(end - time.monotonic(), 0.0)))

    def wait_until(self, condition: Callable[[], bool], timeout: float = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not condition():
            self.check()
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(POLL)
        return True

    def result(self, future, on_cancel: Callable[[], object] = None):
        try:
            self.wait_until(future.done)
        except Canceled:
            if on_cancel is not None:
                on_cancel()
            raise
        return future.result()
