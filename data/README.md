# Data

| File | Source | After cleaning |
| --- | --- | --- |
| `modcloth_final_data.json.gz` | ModCloth | 82,790 transactions · 47,958 users · 1,378 items |
| `renttherunway_final_data.json.gz` | RentTheRunway | 192,544 transactions · 105,571 users · 5,850 items |

Source: [Clothing Fit Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit),
R. Misra, M. Wan, J. McAuley. *Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces.* RecSys 2018.

- Format: JSON Lines, gzip-compressed. `sizerec` reads the `.gz` files directly.
- Label: `fit` ∈ {small, fit, large}; about 70% are `fit`.
- Post-purchase fields (`review_text`, `review_summary`, `rating`, `quality`, `length`) are dropped to prevent leakage.

The data belong to their original authors and are redistributed here only so that the experiments can be reproduced.
