import collections
import re
import threading
import time
from pathlib import Path

QUESTIONS_DIR = Path(__file__).parent / "benchmark_questions"

_RETRY_AFTER_RE      = re.compile(r"(?:(\d+)m)?([\d.]+)s")
_SERVER_ERROR_MARKERS = ("500", "502", "503", "504", "INTERNAL", "unreachable_backend", "Internal server error")
_RETRY_WAIT          = 15


def parse_retry_after(exc: Exception, attempt: int = 1) -> float:
    msg = str(exc)
    for marker in ("try again in ", "retry in ", "retryDelay': '"):
        idx = msg.find(marker)
        if idx != -1:
            m = _RETRY_AFTER_RE.search(msg, idx)
            if m:
                return float(m.group(1) or 0) * 60 + float(m.group(2)) + 5
    base = 120 if any(marker in msg for marker in _SERVER_ERROR_MARKERS) else _RETRY_WAIT
    return min(base * (2 ** (attempt - 1)), 600)


class RateLimiter:
    def __init__(self, max_calls: int = 10, period: float = 60.0):
        self.max_calls = max_calls
        self.period    = period
        self._timestamps: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                sleep_for = self.period - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
