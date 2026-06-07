"""Per-turn conversation statistics for LLM applications.

:class:`ConvStats` records per-turn token counts, cost, and latency then
exposes aggregates (:class:`Stats`) over any slice of the conversation.

Example::

    from llm_conv_stats import ConvStats

    stats = ConvStats()
    stats.record(input_tokens=1024, output_tokens=256, cost_usd=0.02, latency_ms=350)
    stats.record(input_tokens=512,  output_tokens=128, cost_usd=0.01, latency_ms=180)

    s = stats.aggregate()
    print(s.total_input_tokens)   # 1536
    print(s.total_cost_usd)       # 0.03
    print(s.avg_latency_ms)       # 265.0
    print(s.p95_latency_ms)       # 350.0  (max of 2 turns)
    print(s.turns)                # 2
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    """Statistics for a single LLM turn.

    Attributes:
        input_tokens:  Tokens in the prompt.
        output_tokens: Tokens in the completion.
        cost_usd:      Cost in US dollars (``None`` if unknown).
        latency_ms:    Round-trip time in milliseconds (``None`` if unknown).
        label:         Optional human-readable label (e.g. turn number or step name).
        created_at:    Unix timestamp when the record was created.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float | None = None
    label: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "label": self.label,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnRecord:
        """Reconstruct a :class:`TurnRecord` from a plain dict."""
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cost_usd=_optional_float(data.get("cost_usd")),
            latency_ms=_optional_float(data.get("latency_ms")),
            label=str(data.get("label", "")),
            created_at=float(data.get("created_at", 0.0)),
        )

    def __repr__(self) -> str:
        return (
            f"TurnRecord(in={self.input_tokens}, out={self.output_tokens},"
            f" cost={self.cost_usd}, latency_ms={self.latency_ms})"
        )


@dataclass
class Stats:
    """Aggregate statistics over a collection of turns.

    All values are ``None`` when no turns are available for that metric.

    Attributes:
        turns:                Number of turns in the aggregate.
        total_input_tokens:   Sum of input tokens.
        total_output_tokens:  Sum of output tokens.
        total_tokens:         Sum of all tokens.
        total_cost_usd:       Sum of costs (``None`` if no costs recorded).
        avg_cost_usd:         Mean cost per turn (``None`` if no costs recorded).
        min_latency_ms:       Minimum latency (``None`` if no latency recorded).
        max_latency_ms:       Maximum latency (``None`` if no latency recorded).
        avg_latency_ms:       Mean latency (``None`` if no latency recorded).
        p50_latency_ms:       Median latency (``None`` if no latency recorded).
        p95_latency_ms:       95th percentile latency (``None`` if no latency).
        avg_input_tokens:     Mean input tokens per turn.
        avg_output_tokens:    Mean output tokens per turn.
    """

    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float | None = None
    avg_cost_usd: float | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "turns": self.turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_usd": self.avg_cost_usd,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"Stats(turns={self.turns},"
            f" total_tokens={self.total_tokens},"
            f" total_cost_usd={self.total_cost_usd},"
            f" avg_latency_ms={self.avg_latency_ms})"
        )


class ConvStats:
    """Collect and aggregate per-turn LLM statistics.

    Example::

        cs = ConvStats()
        cs.record(input_tokens=500, output_tokens=100, cost_usd=0.01, latency_ms=200)
        cs.record(input_tokens=300, output_tokens=50,  latency_ms=150)

        agg = cs.aggregate()
        print(agg.total_input_tokens)   # 800
        print(agg.avg_latency_ms)       # 175.0
    """

    def __init__(self, *, clock: Any = None) -> None:
        self._turns: list[TurnRecord] = []
        self._clock = clock if clock is not None else time.time

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        label: str = "",
    ) -> TurnRecord:
        """Record statistics for one LLM turn.

        Args:
            input_tokens:  Prompt token count.
            output_tokens: Completion token count.
            cost_usd:      USD cost for this turn (optional).
            latency_ms:    Round-trip latency in ms (optional).
            label:         Optional identifier for this turn.

        Returns:
            The new :class:`TurnRecord`.
        """
        turn = TurnRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            label=label,
            created_at=self._clock(),
        )
        self._turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, *, last_n: int | None = None) -> Stats:
        """Compute aggregate statistics.

        Args:
            last_n: If given, aggregate only the most recent *n* turns. A value
                of ``0`` (or negative) aggregates no turns and returns an empty
                :class:`Stats`.

        Returns:
            A :class:`Stats` instance.
        """
        if last_n is None:
            turns = self._turns
        elif last_n <= 0:
            # ``self._turns[-0:]`` is ``self._turns[0:]`` which returns every
            # turn, so the slice cannot express "the last 0 turns". Handle
            # non-positive counts explicitly to keep the obvious meaning.
            turns = []
        else:
            turns = self._turns[-last_n:]
        n = len(turns)
        if n == 0:
            return Stats()

        total_in = sum(t.input_tokens for t in turns)
        total_out = sum(t.output_tokens for t in turns)

        costs = [t.cost_usd for t in turns if t.cost_usd is not None]
        total_cost = sum(costs) if costs else None
        avg_cost = (sum(costs) / len(costs)) if costs else None

        latencies = sorted(t.latency_ms for t in turns if t.latency_ms is not None)
        if latencies:
            min_lat = latencies[0]
            max_lat = latencies[-1]
            avg_lat = sum(latencies) / len(latencies)
            p50 = _percentile(latencies, 50)
            p95 = _percentile(latencies, 95)
        else:
            min_lat = max_lat = avg_lat = p50 = p95 = None

        return Stats(
            turns=n,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_in + total_out,
            total_cost_usd=total_cost,
            avg_cost_usd=avg_cost,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            avg_input_tokens=total_in / n,
            avg_output_tokens=total_out / n,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def turns(self) -> list[TurnRecord]:
        """Return all recorded turns in insertion order."""
        return list(self._turns)

    def turn_count(self) -> int:
        """Number of recorded turns."""
        return len(self._turns)

    def clear(self) -> None:
        """Remove all recorded turns."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise all turns to a plain dict."""
        return {"turns": [t.to_dict() for t in self._turns]}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, clock: Any = None) -> ConvStats:
        """Reconstruct a :class:`ConvStats` from a plain dict."""
        cs = cls(clock=clock)
        for td in data.get("turns", []):
            turn = TurnRecord.from_dict(td)
            cs._turns.append(turn)
        return cs

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConvStats(turns={len(self._turns)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute *p*-th percentile (nearest rank, 0–100)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = int(math.ceil(p / 100.0 * n)) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]
