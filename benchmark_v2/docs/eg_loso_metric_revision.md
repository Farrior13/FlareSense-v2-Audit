# EG-LOSO Metric Revision: From IoU to IoL

## 1. The Problem with IoU (Temporal Boundary Agreement)
In the initial version (v1) of the EG-LOSO (Event Graph Leave-One-Station-Out) benchmark, event recovery was evaluated using **Intersection over Union (IoU)** of temporal boundaries:

$$ IoU = \frac{A \cap B}{A \cup B} $$

This metric implicitly assumed that a local station's observation ($A$) and the global event graph's predicted cluster ($B$) should have matching durations. However, physical solar radio bursts are global events observed by multiple stations. A global event graph correctly chains overlapping observations into a single continuous event cluster that may span several hours. In contrast, a single station may only record a 15-minute burst.

**Example scenario:**
* **Graph event ($B$)**: 120 minutes
* **Station event ($A$)**: 15 minutes (fully contained within $B$)
* **IoU**: $15 / 120 = 0.125$

Because the threshold was set to `IoU > 0.5`, the metric strictly penalized the global graph *precisely when it successfully reconstructed a long-duration physical event*. The metric was fundamentally too strict for the task of mapping local observations to global events.

## 2. Introducing IoL (Event Coverage)
To correct this methodological error, a new metric was introduced (v2): **Intersection over Local (IoL)** or Event Coverage.

$$ IoL = \frac{Intersection}{Local\ Event\ Duration} $$

IoL specifically answers the question: *Did the global event graph successfully cover the time period of the local observation?* If the graph covers 100% of the local event's duration, the IoL is 1.0, regardless of how long the global event lasts.

Both metrics are preserved for scientific traceability:
* `ERR_IoU_v1`: Measures temporal boundary agreement.
* `ERR_IoL_v2`: Measures event coverage (the primary metric for EG-LOSO).

## 3. Results Comparison (v1 vs v2)

The introduction of IoL revealed that the global event graph successfully covered far more events than IoU suggested. 

| Split Scheme | ERR_IoU_v1 (Old) | ERR_IoL_v2 (New) |
| :--- | :--- | :--- |
| **Random Node Split** (20% drop) | 0.275 | **0.322** |
| **Random Station Split** (20% drop) | 0.266 | **0.311** |
| **EG-LOSO** (Single station drop) | 0.277 | **0.342** |

*Note: For some highly active stations (e.g., `AUSTRIA-UNIGRAZ_01`), the event recovery rate jumped from 36.9% (IoU) to 48.3% (IoL).*

## 4. Interpretation

The results lead to a strong, neutral conclusion:
**The event graph correctly restores event coverage, but in the current EG-LOSO formulation, it does not show a substantial advantage over random station dropout schemes.**

Key takeaways:
1. **Network Redundancy is the Primary Driver:** Removing a single critical station (EG-LOSO) degrades performance to a similar degree as removing a random 20% of all stations globally. This physically validates the architecture of the e-CALLISTO network—there is no single critical point of failure, and the sensory information is highly distributed and redundant.
2. **Methodological Validity over ML "Magic":** The event graph's value lies not in artificially inflating ML predictive metrics, but in **structuring events and controlling for data leakage**. It provides a scientifically grounded framework for understanding multi-station observations, which is exactly why it was built for this audit.

This revision stands as a critical self-correction in the audit process: finding a methodological flaw in our own analysis, correcting it, and arriving at a physically sound and honest conclusion.
