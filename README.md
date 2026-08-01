# Independent Reproducibility Audit of FlareSense-v2

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No GPU Required](https://img.shields.io/badge/GPU-Not%20Required-green.svg)]()
[![Data: HuggingFace](https://img.shields.io/badge/Data-HuggingFace-orange.svg)](https://huggingface.co/datasets/i4ds/ecallisto_radio_sunburst)

An independent methodological audit of **FlareSense-v2** (Timmel et al. 2026, [arXiv:2607.26014v1](https://arxiv.org/html/2607.26014v1)), a deep learning system for solar radio burst detection on e-CALLISTO spectrograms.

We identify event-level data leakage (65–74% of test bursts share physical events with training), test-set hyperparameter tuning, and base rate sensitivity. After removing event overlap, **recall drops from 83.4% to 73.3%** (−10 pp) and model confidence shifts (median 0.978 → 0.831). The headline precision drop of −15.4 pp includes a compositional artifact from the changed class balance (see full report §4.1). All findings are reproducible from the publicly available HuggingFace dataset without GPU access.

> **⚠️ Model version note:** The predictions in the HuggingFace dataset were uploaded on 19 October 2024, before the FlareSense-v2 checkpoint (January 2025) and paper (July 2026). Structural findings (event-level leakage, test-set tuning) apply directly; specific delta magnitudes may differ for the final model. See the [full report](docs/audit_v1.md) for details.

## Key Findings

1. **Event-Level Data Leakage**: 65–74% of test burst samples share a physical solar event with training data. The random split does not account for the fact that the same burst is recorded by multiple stations simultaneously.

2. **Performance Degradation**: Removing event overlap drops recall from 83.4% to 73.3% (−10.0 pp on burst samples) and shifts model confidence (median prob 0.978 → 0.831). The headline precision drop (−15.4 pp) and F1 drop (−10.7 pp) are statistically significant (bootstrap 95% CI excludes zero) but include a compositional component from the changed class balance in the clean subset.

3. **Confidence Shift**: The model assigns systematically higher probabilities to leaked bursts (median 0.978) than clean ones (median 0.831), indicating event-specific calibration rather than physics-based generalization.

4. **Test-Set Hyperparameter Tuning**: A 6-link trace through the repository shows that hyperparameter optimization targeted `test_avg_f1` with `val_split: test`, contradicting Section 5.2 of the paper.

5. **Base Rate Sensitivity**: At realistic deployment prevalences (0.1–1%), precision drops to 6–39% by Bayes' theorem, undiscussed in the paper's deployment claims.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce all metrics (downloads ~67.5GB dataset with images on first run)
python scripts/reproduce_all.py

# Generate all 6 figures
python scripts/generate_figures.py

# Find concrete leakage examples
python scripts/leakage_examples.py

# Per-station failure mode analysis
python scripts/per_station_analysis.py
```

## Repository Structure

```
FlareSense-v2-Audit/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── audit_v1.md              # Full audit report (v1.1)
├── scripts/
│   ├── reproduce_all.py          # Main: reproduces all metrics + bootstrap CIs
│   ├── generate_figures.py       # Generates all 6 publication-ready figures
│   ├── leakage_examples.py       # Concrete event-level leakage examples
│   └── per_station_analysis.py   # Per-station metrics + confidence analysis
├── figures/
│   ├── fig1_negative_prob_dist.png
│   ├── fig2_burst_vs_nonburst.png
│   ├── fig3_base_rate_ppv.png
│   ├── fig4_metrics_comparison.png
│   ├── fig5_leaked_vs_clean_prob.png
│   └── fig6_per_station_delta_f1.png
└── results/                      # Generated JSON outputs
```

## Full Report

📄 **[Read the full audit report →](docs/audit_v1.md)**

## Citation

```bibtex
@misc{flaresense_audit_2026,
  title   = {Independent Reproducibility Audit of {FlareSense}-v2:
             Evidence of Event-Level Evaluation Leakage in
             Solar Radio Burst Classification},
  year    = {2026},
  note    = {Version 1.1},
  url     = {https://github.com/Farrior13/FlareSense-v2-Audit}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This audit uses the publicly available dataset [`i4ds/ecallisto_radio_sunburst`](https://huggingface.co/datasets/i4ds/ecallisto_radio_sunburst) and references the code repository [`i4Ds/FlareSense-v2`](https://github.com/i4Ds/FlareSense-v2). We thank the original authors for making their data and code publicly available.
