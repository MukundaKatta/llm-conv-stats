# llm-conv-stats

Per-turn conversation statistics for LLM applications.

Record token counts, cost, and latency for each LLM call, then aggregate them into min/max/avg/p50/p95 stats over the full conversation or any recent slice.

## Install

```bash
pip install llm-conv-stats
```

## Quick start

```python
from llm_conv_stats import ConvStats

stats = ConvStats()
stats.record(input_tokens=1024, output_tokens=256, cost_usd=0.02, latency_ms=350)
stats.record(input_tokens=512,  output_tokens=128, cost_usd=0.01, latency_ms=180)

s = stats.aggregate()
print(s.turns)               # 2
print(s.total_input_tokens)  # 1536
print(s.total_cost_usd)      # 0.03
print(s.avg_latency_ms)      # 265.0
print(s.p95_latency_ms)      # 350.0

# Last 5 turns only
recent = stats.aggregate(last_n=5)
```

## API

### `ConvStats`

| Method | Description |
|---|---|
| `record(*, input_tokens, output_tokens, cost_usd, latency_ms, label)` | Record one turn |
| `aggregate(*, last_n=None)` | Compute `Stats` over all (or last n) turns |
| `turns()` | List of `TurnRecord` objects |
| `turn_count()` | Number of turns recorded |
| `clear()` | Remove all turns |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

### `Stats`

| Field | Type | Description |
|---|---|---|
| `turns` | `int` | Turn count |
| `total_input_tokens` | `int` | Sum of input tokens |
| `total_output_tokens` | `int` | Sum of output tokens |
| `total_tokens` | `int` | Sum of all tokens |
| `total_cost_usd` | `float \| None` | Total cost |
| `avg_cost_usd` | `float \| None` | Mean cost per turn |
| `min_latency_ms` | `float \| None` | Minimum latency |
| `max_latency_ms` | `float \| None` | Maximum latency |
| `avg_latency_ms` | `float \| None` | Mean latency |
| `p50_latency_ms` | `float \| None` | Median latency |
| `p95_latency_ms` | `float \| None` | 95th percentile latency |
| `avg_input_tokens` | `float` | Mean input tokens per turn |
| `avg_output_tokens` | `float` | Mean output tokens per turn |

## License

MIT
