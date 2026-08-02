# Station Contribution Analysis

This analysis measures the resilience of the network when a specific station is removed, compared to a random 20% station dropout. A positive contribution means the network predicts this station better than it predicts a random subset (the station is highly redundant and well-covered by others). A negative contribution means the station contains unique observations that the rest of the network struggles to reconstruct.

**Baseline (Random Station 20% Drop) IoL:** 0.348

### Top Contributors (Highly Redundant / Well Covered)
| Station | IoL | Contribution |
| :--- | :--- | :--- |
| AUSTRIA-UNIGRAZ_01 | 0.483 | +0.135 |
| EGYPT-Alexandria_02 | 0.476 | +0.128 |
| HUMAIN_59 | 0.418 | +0.071 |
| MRO_61 | 0.414 | +0.066 |
| NORWAY-EGERSUND_01 | 0.400 | +0.052 |
| BIR_01 | 0.382 | +0.035 |
| SWISS-Landschlacht_62 | 0.381 | +0.034 |
| GERMANY-DLR_63 | 0.375 | +0.027 |
| INDIA-OOTY_02 | 0.366 | +0.018 |
| SSRT_59 | 0.348 | +0.000 |

### Weak Contributors (Unique / Poorly Covered)
| Station | IoL | Contribution |
| :--- | :--- | :--- |
| GLASGOW_01 | 0.321 | -0.026 |
| ALASKA-COHOE_63 | 0.211 | -0.137 |
| ALASKA-HAARP_62 | 0.199 | -0.148 |
| USA-ARIZONA-ERAU_01 | 0.183 | -0.164 |
| Australia-ASSA_62 | 0.179 | -0.169 |
