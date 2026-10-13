# Storage Benchmark (A10)
Generated over 5 median iterations. Warm cache assumed for SQL reads.

| Format                     |   Size_MB |   Write_Sec |   Read_Sec |
|:---------------------------|----------:|------------:|-----------:|
| CSV                        |     49.37 |     16.6064 |     2.3732 |
| JSON                       |     95.76 |      4.066  |     6.508  |
| Parquet                    |     10.85 |      1.2201 |     0.3704 |
| PostgreSQL (Warm)          |     43.06 |     26.9006 |     6.318  |
| PostgreSQL (Filtered High) |     43.06 |     26.9006 |     0.3484 |

### Recommendations:
* **Source Exchange:** CSV (highly compatible, immutable).
* **Intermediate Storage:** Parquet (fastest I/O, preserves datatypes, smallest size).
* **Model Input:** PostgreSQL (enforces primary keys, allows filtered reads).
* **Analytics:** PostgreSQL or Parquet based on BI tool connectivity.
