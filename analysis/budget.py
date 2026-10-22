"""Hard daily Firestore budget shared by all components (doc `meta/budget`, see SCHEMA.md).

Protocol: read `meta/budget` once at run start (skip the run when the analysis share or the
combined limit is used up), stop gracefully when the remaining budget is reached (keeping a
reserve for the aggregation of what was processed), and add the run's usage in one write at
the end (increment on the same day, reset on a new Berlin day; optimistic precondition).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from .firestore import Write
from .textutil import today_berlin

log = logging.getLogger(__name__)

BUDGET_DOC = "meta/budget"
COMPONENT = "analysis"
COMPONENTS = ("crawler", "analysis", "api")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class Budget:
    """Tracks this run's Firestore usage (from the client's read/write counters) against the day."""

    def __init__(
        self,
        fs,
        doc=None,
        today: str | None = None,
        *,
        write_limit: int | None = None,
        read_limit: int | None = None,
        share_writes: int | None = None,
        share_reads: int | None = None,
        reserve_writes: int | None = None,
        reserve_reads: int | None = None,
    ):
        self.fs = fs
        self.today = today or today_berlin()
        self.write_limit = write_limit if write_limit is not None else _env_int("DAILY_WRITE_LIMIT", 60000)
        self.read_limit = read_limit if read_limit is not None else _env_int("DAILY_READ_LIMIT", 200000)
        self.share_writes = share_writes if share_writes is not None else _env_int("ANALYSIS_DAILY_WRITE_LIMIT", 30000)
        self.share_reads = share_reads if share_reads is not None else _env_int("ANALYSIS_DAILY_READ_LIMIT", 120000)
        # kept back for the aggregation of whatever this run processed
        self.reserve_writes = (
            reserve_writes if reserve_writes is not None else _env_int("ANALYSIS_AGG_RESERVE_WRITES", 2000)
        )
        self.reserve_reads = (
            reserve_reads if reserve_reads is not None else _env_int("ANALYSIS_AGG_RESERVE_READS", 10000)
        )
        self._doc = doc
        data = (doc.data if doc else None) or {}
        same_day = data.get("day") == self.today
        w = (data.get("writes") or {}) if same_day else {}
        r = (data.get("reads") or {}) if same_day else {}
        self.day_writes = {c: int(w.get(c) or 0) for c in COMPONENTS}
        self.day_reads = {c: int(r.get(c) or 0) for c in COMPONENTS}
        self._w0 = fs.writes
        self._r0 = fs.reads
        self.recorded = False

    @classmethod
    def load(cls, fs, **kw) -> Budget:
        doc = fs.get(BUDGET_DOC)  # 1 read (counted in this run's usage)
        b = cls(fs, doc, **kw)
        b._r0 -= 1
        return b

    # ------------------------------------------------------------------ accounting
    @property
    def run_writes(self) -> int:
        return self.fs.writes - self._w0

    @property
    def run_reads(self) -> int:
        return self.fs.reads - self._r0

    def remaining_writes(self) -> int:
        own = self.share_writes - self.day_writes[COMPONENT]
        total = self.write_limit - sum(self.day_writes.values())
        return min(own, total) - self.run_writes

    def remaining_reads(self) -> int:
        own = self.share_reads - self.day_reads[COMPONENT]
        total = self.read_limit - sum(self.day_reads.values())
        return min(own, total) - self.run_reads

    def exhausted(self) -> str | None:
        """Why a run must not start (None when it may)."""
        if self.remaining_writes() <= self.reserve_writes:
            return f"write budget used up ({self.remaining_writes()} left, reserve {self.reserve_writes})"
        if self.remaining_reads() <= self.reserve_reads:
            return f"read budget used up ({self.remaining_reads()} left, reserve {self.reserve_reads})"
        return None

    def allows(self, writes: int = 0, reads: int = 0) -> bool:
        """Whether a step with this estimated usage fits and still leaves the aggregation reserve."""
        return (
            self.remaining_writes() - writes >= self.reserve_writes
            and self.remaining_reads() - reads >= self.reserve_reads
        )

    def summary(self) -> dict:
        return {
            "day": self.today,
            "runWrites": self.run_writes,
            "runReads": self.run_reads,
            "remainingWrites": self.remaining_writes(),
            "remainingReads": self.remaining_reads(),
        }

    # ------------------------------------------------------------------ end of run
    def record(self, retries: int = 3) -> None:
        """Add this run's usage to `meta/budget` in one write (+1 re-read per conflict)."""
        if self.recorded:
            return
        doc = self._doc
        for attempt in range(retries):
            writes = self.run_writes + 1  # including this write
            reads = self.run_reads
            today = today_berlin()
            ts = datetime.now(UTC)
            data = (doc.data if doc else None) or {}
            if doc is not None and data.get("day") == today:
                w = Write(
                    BUDGET_DOC,
                    {"updatedAt": ts},
                    mask=["updatedAt"],
                    exists=True,
                    increments={f"writes.{COMPONENT}": writes, f"reads.{COMPONENT}": reads},
                )
            else:
                # new day (or first use): reset all counters, guarded against a concurrent reset
                w = Write(
                    BUDGET_DOC,
                    {
                        "day": today,
                        "writes": {c: (writes if c == COMPONENT else 0) for c in COMPONENTS},
                        "reads": {c: (reads if c == COMPONENT else 0) for c in COMPONENTS},
                        "updatedAt": ts,
                    },
                    update_time=doc.update_time if doc is not None else None,
                    exists=False if doc is None else None,
                )
            try:
                self.fs.commit_atomic([w])
                self.recorded = True
                log.info("budget: recorded %d writes, %d reads for %s", writes, reads, today)
                return
            except RuntimeError as e:
                msg = str(e)
                if not any(k in msg for k in ("FAILED_PRECONDITION", "ALREADY_EXISTS", "precondition", "409")):
                    raise
                log.warning("budget doc changed concurrently, re-reading (attempt %d)", attempt + 1)
                doc = self.fs.get(BUDGET_DOC)
        log.error("budget: could not record usage after %d attempts", retries)
