# Ablation Ladder & Final Statistical Analysis

## 1. Graph Structural Stats (Constant across IoL Evaluation Thresholds)

*Note: The structural graph is built based on physical 15m temporal overlap. The threshold evaluated (0.3, 0.5, etc.) is the strictness of the boundary matching (`IoL`), so the graph structure itself is constant across these evaluations.*

- **Edges**: 9,045
- **Components**: 21,372
- **Mean component size**: 1.35 nodes
- **Largest component fraction (LCF)**: 0.00028 (0.028%)

**Interpretation**: The graph is incredibly sparse. It does NOT collapse into a giant component (LCF is practically zero). Most events are tiny, localized clusters (1.35 nodes on average), confirming that the leakage isolation is extremely robust.

---

## 2. Statistical Significance (Delta IoL + Bootstrap CI)

We paired the `Graph` vs `Temporal` recovery for each of the top 15 stations and bootstrapped the mean difference over 10,000 station-level folds.

| Threshold | $\Delta IoL$ (Graph - Temporal) | 95% Bootstrap CI |
| :--- | :--- | :--- |
| **0.3** | +0.0090 (0.90%) | [+0.0049, +0.0134] |
| **0.5** | +0.0123 (1.23%) | [+0.0088, +0.0169] |
| **0.7** | +0.0153 (1.53%) | [+0.0109, +0.0198] |
| **0.9** | +0.0171 (1.71%) | [+0.0127, +0.0213] |

**Interpretation**: The confidence interval **does not** include zero, meaning the graph *is* statistically significantly better than raw temporal coincidence. However, the **effect size is incredibly marginal** (~1.2% absolute improvement). The graph topology does provide a mathematically real advantage, but physically, it's virtually identical to raw temporal coincidence.

---

## 3. Ablation Matrix (Threshold = 0.5)

| Split | Temporal Recovery | Graph Recovery |
| :--- | :--- | :--- |
| **Random station drop** | 0.323 (`T-Random`) | 0.328 (`G-Random`) |
| **Targeted EG-LOSO drop** | 0.330 (`T-EG`) | 0.342 (`G-EG`) |

### Decomposing the Effects:

1. **Does the graph help on average? (`G-Random` vs `T-Random`)**
   - $0.328 - 0.323 = +0.005$
   - Marginal structural gain.

2. **Does the graph help for critical isolated stations? (`G-EG` vs `T-EG`)**
   - $0.342 - 0.330 = +0.012$
   - Slightly higher structural gain when dealing with highly active/critical nodes, but still marginal.

3. **Does the network naturally compensate for critical nodes using raw time? (`T-EG` vs `T-Random`)**
   - $0.330 - 0.323 = +0.007$
   - The temporal overlap baseline itself scores higher on critical stations than random stations, indicating that critical stations observe major events that are naturally seen by many other stations concurrently.

4. **Overall impact of EG-LOSO split (`G-EG` vs `G-Random`)**
   - $0.342 - 0.328 = +0.014$
   - The full evaluation protocol yields a slightly higher IoL because the most critical stations are deeply embedded in redundant, major solar events.
