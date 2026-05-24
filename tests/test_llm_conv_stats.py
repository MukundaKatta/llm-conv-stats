"""Tests for llm-conv-stats."""

from __future__ import annotations

import pytest

from llm_conv_stats import ConvStats, Stats, TurnRecord

# ---------------------------------------------------------------------------
# TurnRecord
# ---------------------------------------------------------------------------


def test_turn_record_defaults():
    t = TurnRecord()
    assert t.input_tokens == 0
    assert t.output_tokens == 0
    assert t.cost_usd is None
    assert t.latency_ms is None
    assert t.label == ""


def test_turn_record_total_tokens():
    t = TurnRecord(input_tokens=100, output_tokens=50)
    assert t.total_tokens == 150


def test_turn_record_to_dict():
    t = TurnRecord(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
        latency_ms=200.0,
        label="turn-1",
        created_at=1000.0,
    )
    d = t.to_dict()
    assert d["input_tokens"] == 10
    assert d["output_tokens"] == 5
    assert d["cost_usd"] == pytest.approx(0.01)
    assert d["latency_ms"] == pytest.approx(200.0)
    assert d["label"] == "turn-1"
    assert d["created_at"] == 1000.0


def test_turn_record_from_dict_round_trip():
    t = TurnRecord(
        input_tokens=50,
        output_tokens=20,
        cost_usd=0.005,
        latency_ms=99.9,
        label="step",
        created_at=500.0,
    )
    restored = TurnRecord.from_dict(t.to_dict())
    assert restored.input_tokens == t.input_tokens
    assert restored.output_tokens == t.output_tokens
    assert restored.cost_usd == pytest.approx(t.cost_usd)
    assert restored.latency_ms == pytest.approx(t.latency_ms)
    assert restored.label == t.label


def test_turn_record_from_dict_none_fields():
    t = TurnRecord.from_dict({"input_tokens": 5})
    assert t.cost_usd is None
    assert t.latency_ms is None


def test_turn_record_repr():
    t = TurnRecord(input_tokens=100, output_tokens=50)
    r = repr(t)
    assert "100" in r
    assert "50" in r


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


def test_stats_defaults():
    s = Stats()
    assert s.turns == 0
    assert s.total_tokens == 0
    assert s.total_cost_usd is None


def test_stats_to_dict():
    s = Stats(turns=2, total_input_tokens=100, total_output_tokens=50)
    d = s.to_dict()
    assert d["turns"] == 2
    assert d["total_input_tokens"] == 100


def test_stats_repr():
    s = Stats(turns=3, total_tokens=500)
    r = repr(s)
    assert "Stats" in r
    assert "3" in r


# ---------------------------------------------------------------------------
# ConvStats — record
# ---------------------------------------------------------------------------


def _clock(start: float = 0.0):
    t = [start]

    def _c():
        val = t[0]
        t[0] += 1.0
        return val

    return _c


def test_record_returns_turn():
    cs = ConvStats(clock=_clock())
    turn = cs.record(input_tokens=100, output_tokens=50)
    assert isinstance(turn, TurnRecord)
    assert turn.input_tokens == 100


def test_record_optional_fields():
    cs = ConvStats()
    turn = cs.record(
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.02,
        latency_ms=300.0,
        label="step-1",
    )
    assert turn.cost_usd == pytest.approx(0.02)
    assert turn.latency_ms == pytest.approx(300.0)
    assert turn.label == "step-1"


def test_turn_count():
    cs = ConvStats()
    cs.record(input_tokens=10)
    cs.record(input_tokens=20)
    assert cs.turn_count() == 2
    assert len(cs) == 2


def test_turns_returns_list():
    cs = ConvStats()
    cs.record(input_tokens=5)
    turns = cs.turns()
    assert len(turns) == 1
    assert isinstance(turns[0], TurnRecord)


def test_clear():
    cs = ConvStats()
    cs.record(input_tokens=100)
    cs.clear()
    assert cs.turn_count() == 0


# ---------------------------------------------------------------------------
# ConvStats — aggregate
# ---------------------------------------------------------------------------


def test_aggregate_empty():
    cs = ConvStats()
    s = cs.aggregate()
    assert s.turns == 0
    assert s.total_tokens == 0
    assert s.total_cost_usd is None
    assert s.avg_latency_ms is None


def test_aggregate_total_tokens():
    cs = ConvStats()
    cs.record(input_tokens=100, output_tokens=50)
    cs.record(input_tokens=200, output_tokens=80)
    s = cs.aggregate()
    assert s.total_input_tokens == 300
    assert s.total_output_tokens == 130
    assert s.total_tokens == 430


def test_aggregate_avg_tokens():
    cs = ConvStats()
    cs.record(input_tokens=100, output_tokens=50)
    cs.record(input_tokens=200, output_tokens=50)
    s = cs.aggregate()
    assert s.avg_input_tokens == pytest.approx(150.0)
    assert s.avg_output_tokens == pytest.approx(50.0)


def test_aggregate_cost():
    cs = ConvStats()
    cs.record(cost_usd=0.01)
    cs.record(cost_usd=0.03)
    s = cs.aggregate()
    assert s.total_cost_usd == pytest.approx(0.04)
    assert s.avg_cost_usd == pytest.approx(0.02)


def test_aggregate_no_cost_is_none():
    cs = ConvStats()
    cs.record(input_tokens=100)
    s = cs.aggregate()
    assert s.total_cost_usd is None
    assert s.avg_cost_usd is None


def test_aggregate_partial_cost():
    # Only some turns have cost — only those count
    cs = ConvStats()
    cs.record(cost_usd=0.02)
    cs.record()  # no cost
    s = cs.aggregate()
    assert s.total_cost_usd == pytest.approx(0.02)
    assert s.avg_cost_usd == pytest.approx(0.02)


def test_aggregate_latency():
    cs = ConvStats()
    cs.record(latency_ms=100.0)
    cs.record(latency_ms=200.0)
    cs.record(latency_ms=300.0)
    s = cs.aggregate()
    assert s.min_latency_ms == pytest.approx(100.0)
    assert s.max_latency_ms == pytest.approx(300.0)
    assert s.avg_latency_ms == pytest.approx(200.0)


def test_aggregate_no_latency_is_none():
    cs = ConvStats()
    cs.record(input_tokens=100)
    s = cs.aggregate()
    assert s.avg_latency_ms is None
    assert s.p50_latency_ms is None
    assert s.p95_latency_ms is None


def test_aggregate_single_turn():
    cs = ConvStats()
    cs.record(input_tokens=50, output_tokens=25, latency_ms=150.0)
    s = cs.aggregate()
    assert s.turns == 1
    assert s.p50_latency_ms == pytest.approx(150.0)
    assert s.p95_latency_ms == pytest.approx(150.0)


def test_aggregate_p95_latency():
    cs = ConvStats()
    for i in range(1, 21):  # 20 turns: 100, 200, ..., 2000
        cs.record(latency_ms=float(i * 100))
    s = cs.aggregate()
    # p95 of 20 values = index ceil(0.95*20)-1 = ceil(19)-1 = 18 → 1900.0
    assert s.p95_latency_ms == pytest.approx(1900.0)


def test_aggregate_p50_latency():
    cs = ConvStats()
    for v in [100.0, 200.0, 300.0]:
        cs.record(latency_ms=v)
    s = cs.aggregate()
    # p50 of 3 = index ceil(1.5)-1 = 1 → 200.0
    assert s.p50_latency_ms == pytest.approx(200.0)


def test_aggregate_last_n():
    cs = ConvStats()
    cs.record(input_tokens=100)
    cs.record(input_tokens=200)
    cs.record(input_tokens=300)
    s = cs.aggregate(last_n=2)
    assert s.turns == 2
    assert s.total_input_tokens == 500  # 200 + 300


def test_aggregate_last_n_larger_than_turns():
    cs = ConvStats()
    cs.record(input_tokens=50)
    s = cs.aggregate(last_n=10)
    assert s.turns == 1


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_round_trip():
    cs = ConvStats(clock=_clock())
    cs.record(input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=200.0)
    cs.record(input_tokens=200, output_tokens=80, latency_ms=150.0)

    restored = ConvStats.from_dict(cs.to_dict(), clock=_clock())
    assert restored.turn_count() == 2
    s = restored.aggregate()
    assert s.total_input_tokens == 300
    assert s.total_cost_usd == pytest.approx(0.01)


def test_from_dict_empty():
    cs = ConvStats.from_dict({})
    assert cs.turn_count() == 0


def test_repr():
    cs = ConvStats()
    cs.record(input_tokens=10)
    assert "ConvStats" in repr(cs)
    assert "1" in repr(cs)
