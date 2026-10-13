# Storage Benchmark (A10)
Generated over 5 median iterations. Warm cache assumed for SQL reads.

| Format                     |   Size_MB |   Write_Sec |   Read_Sec |
|:---------------------------|----------:|------------:|-----------:|
| CSV                        |     49.46 |      3.9122 |     1.0769 |
| JSON                       |     95.76 |      2.3693 |     3.55   |
| Parquet                    |     10.81 |      0.5965 |     0.3249 |
| PostgreSQL (Warm)          |     42.03 |     25.2525 |     5.0976 |
| PostgreSQL (Filtered High) |     42.03 |     25.2525 |     0.2013 |

### Recommendations:
* **Source Exchange:** CSV (highly compatible, immutable).
* **Intermediate Storage:** Parquet (fastest I/O, preserves datatypes, smallest size).
* **Model Input:** PostgreSQL (enforces primary keys, allows filtered reads).
* **Analytics:** PostgreSQL or Parquet based on BI tool connectivity.
