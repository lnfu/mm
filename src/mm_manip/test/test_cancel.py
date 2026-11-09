import threading
import time

import pytest

from mm_manip.cancel import CancelToken
from mm_manip.errors import Canceled


class Future:
    def __init__(self, value=None):
        self.value = value
        self._done = False

    def done(self):
        return self._done

    def result(self):
        return self.value


def test_default_token_is_never_canceled():
    token = CancelToken()
    token.check()
    assert not token.canceled


def test_check_raises_once_canceled():
    flag = []
    token = CancelToken(lambda: bool(flag))
    token.check()
    flag.append(1)
    with pytest.raises(Canceled):
        token.check()


def test_sleep_ends_early_on_cancel():
    start = time.monotonic()
    token = CancelToken(lambda: time.monotonic() - start > 0.05)
    with pytest.raises(Canceled):
        token.sleep(5.0)
    assert time.monotonic() - start < 1.0


def test_wait_until_true_false_and_canceled():
    token = CancelToken()
    assert token.wait_until(lambda: True, timeout=0.0)
    assert not token.wait_until(lambda: False, timeout=0.05)
    with pytest.raises(Canceled):
        CancelToken(lambda: True).wait_until(lambda: False)


def test_result_waits_for_the_future():
    future = Future("done")
    threading.Timer(0.05, lambda: setattr(future, "_done", True)).start()
    assert CancelToken().result(future) == "done"


def test_result_runs_on_cancel_before_raising():
    stopped = []
    with pytest.raises(Canceled):
        CancelToken(lambda: True).result(Future(), on_cancel=lambda: stopped.append(1))
    assert stopped == [1]
