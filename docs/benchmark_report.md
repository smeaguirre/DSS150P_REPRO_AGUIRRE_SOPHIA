# Storage and Partitioning Benchmark Report

## Storage Format Comparison
| Format     |   Size (MB) |   Write Time (s) |   Median Read Time (s) |
|:-----------|------------:|-----------------:|-----------------------:|
| CSV        |     41.3355 |         2.81696  |              0.789753  |
| JSONL      |     82.2509 |         3.22004  |              2.54185   |
| Parquet    |     10.8176 |         0.739158 |              0.0320671 |
| PostgreSQL |    nan      |        24.9197   |              2.21449   |

**PostgreSQL Specifics:**
- Full Table Query Median Time: 2.2145 s
- Filtered Query (`popularity_tier = 'High'`) Median Time: 0.1386 s

## Conclusion
Parquet is the optimal format for this Spotify dataset during intermediate pipeline stages. It offers the strongest compression and fastest disk read/write speeds natively in Python. CSV and JSON require substantially more space and serialization overhead. PostgreSQL provides analytical querying power and API backend utility but incurs network and protocol overhead for full-table scans compared to Parquet.
