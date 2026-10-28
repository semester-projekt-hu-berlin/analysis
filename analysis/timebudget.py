"""Soft time budget of a run: stop cleanly before the CI job timeout kills the process.

GitHub cancels the analysis job after `timeout-minutes` (150). A run that is killed while it
continues a long job (relink, rescore) keeps its progress only up to the last saved cursor and
leaves the (entity, day) pairs it wrote in `pendingPairs` unaggregated until the next run. With a
time budget the main loop and the one-off jobs check the clock before each step (a batch, a
page, a group of pairs); when the budget is reached they save their cursor, aggregate what they
processed and return with `stopped`, and the run records its Firestore usage as usual.

    ANALYSIS_TIME_BUDGET_MIN  minutes the run may take (default 135; 0 = no limit)
    ANALYSIS_JOB_START        Unix time the CI job started (set by analyze.yml), so the
                              budget also covers the setup steps; default: process start
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

DEFAULT_MINUTES = 135.0  # analyze.yml: timeout-minutes 150, minus the time a final aggregation needs


def _env_float(name: str) -> float | None:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else None
    except ValueError:
        return None


class TimeBudget:
    def __init__(self, minutes: float | None = None, start: float | None = None, clock=time.time):
        self.clock = clock
        if minutes is None:
            minutes = _env_float("ANALYSIS_TIME_BUDGET_MIN")
            minutes = DEFAULT_MINUTES if minutes is None else minutes
        if start is None:
            start = _env_float("ANALYSIS_JOB_START") or clock()
        self.minutes = minutes
        self.start = start
        self.deadline = start + minutes * 60 if minutes and minutes > 0 else None
        self.hit: str | None = None

    def left(self) -> float:
        """Seconds left (infinite without a limit)."""
        return float("inf") if self.deadline is None else self.deadline - self.clock()

    def reached(self, what: str = "", reserve: float = 0.0) -> bool:
        """True when less than `reserve` seconds (the expected duration of the next step) are
        left. Logs one "time budget reached" line per run."""
        if self.deadline is None or self.left() > reserve:
            return False
        if self.hit is None:
            self.hit = what or "run"
            log.warning(
                "time budget reached (%.0f of %.0f min used, next step needs ~%.0f s): stopping %s cleanly",
                (self.clock() - self.start) / 60,
                self.minutes,
                reserve,
                self.hit,
            )
        return True

    def summary(self) -> dict:
        return {
            "minutes": self.minutes,
            "usedMinutes": round((self.clock() - self.start) / 60, 1),
            "reached": self.hit,
        }


_current: TimeBudget | None = None


def get() -> TimeBudget:
    """The process-wide time budget (created from the environment on first use)."""
    global _current
    if _current is None:
        _current = TimeBudget()
    return _current


def set_current(tb: TimeBudget | None) -> None:
    """Install a time budget (the CLI at start, tests); None resets to the environment default."""
    global _current
    _current = tb
