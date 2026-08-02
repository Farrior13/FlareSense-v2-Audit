### 1. Purpose

> This benchmark evaluates whether the event graph structure is non-trivial, stable, and robust against randomization.

### 2. Что проверяем

| Test                 | Question                                                      |
| -------------------- | ------------------------------------------------------------- |
| Temporal sensitivity | Меняется ли граф физически ожидаемым образом при разных окнах |
| Null Model 1         | Превышает ли реальная структура случайную временную структуру |
| Configuration model  | Объясняется ли граф только степенным распределением           |
| Station shuffle      | Есть ли межстанционный сигнал                                 |
| LOSO ablation        | Есть ли зависимость от отдельных станций                      |
| Event isolation      | Есть ли event leakage внутри графа                            |

### 3. Важное ограничение

> This benchmark is not used as evidence for the primary audit findings. The main conclusions are based on leakage reproduction and repository forensics.

---

### 4. Architecture

All modules share a single canonical graph construction function via `graph_utils.py`:

```
graph_utils.py             ← Single source of truth
  ├── build_graph()          - temporal overlap graph
  ├── build_graph_events()   - graph + per-row component labels
  ├── compute_temporal_iou() - IoU metric
  ├── compute_event_coverage() - IoL metric
  └── evaluate_topology()   - GCF, component count
        │
        ├── benchmark_core.py    ← Main audit protocol (Sensitivity, Null, LOSO, Isolation)
        ├── eg_loso_core.py      ← EG-LOSO evaluation (Graph vs Temporal baseline)
        ├── structural_core.py   ← Targeted edge removal, event profiles
        ├── adversarial_core.py  ← Null models (Degree, ER), window sweep, edge perturbation
        ├── temporal_null_core.py← Time-shifting null models, pruning
        └── coverage_analysis.py ← Correlation: ERR vs network topology
```

**Metric naming**:
- `ClusterPurity` (benchmark_core.py) — fraction of burst nodes in active clusters
- `MatchPurity` (adversarial_core.py) — fraction of matched events matching physical events
- `IoL` / `Event Coverage` — Intersection over Local duration
- `IoU` — Intersection over Union

### 5. Running

```bash
# Unit tests (no dataset required)
python test_metrics.py

# Full benchmark (requires HuggingFace dataset)
python benchmark_core.py --split test

# Individual modules
python eg_loso_core.py --split test
python structural_core.py
python adversarial_core.py
python temporal_null_core.py
python coverage_analysis.py

# Stress test
python event_graph/falsification/stress_test_graph.py
```

### 6. Results Status

See [docs/RESULTS_STATUS.md](docs/RESULTS_STATUS.md) for the current state of CSV results
and their relationship to the fixed code.
