# Independent Reproducibility Audit of FlareSense-v2

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: HuggingFace](https://img.shields.io/badge/Data-HuggingFace-orange.svg)](https://huggingface.co/datasets/i4ds/ecallisto_radio_sunburst)
[![Preprint: arXiv](https://img.shields.io/badge/arXiv-2607.26014-b31b1b.svg)](https://arxiv.org/abs/2607.26014)

This repository contains the complete independent evaluation audit, controlled retraining framework, and replication package for **FlareSense-v2** (Vincenzo Timmel, André Csillaghy, Christian Monstein, 2026, [arXiv:2607.26014v1](https://arxiv.org/abs/2607.26014)), an automated ResNet-34 system for solar radio burst detection on global e-CALLISTO radio spectrograms.

---

## Executive Summary

FlareSense-v2 reported headline test performance of 93.0% precision and 73.15% recall. Our independent reproducibility audit reveals four primary findings:

1. **Multi-Station Event Leakage (Observational Disparity)**: Because spectrograms were partitioned randomly by file rather than by astronomical UTC event window, **65.6% of test bursts** (2,691 of 4,100) share an identical 15-minute event window with training samples (expanding to 74.3% within $\pm 60$ minutes). On truly unseen ("clean") solar events, test recall drops from **88.15% to 42.87%** (an observational deficit of **45.28 percentage points**, 95% CI: [39.4, 51.2]). Median model confidence drops from **0.790 to 0.289**, falling well below the 0.426 classification threshold.
2. **Asymmetric Annotation Protocol**: Cross-matching all 304,750 dataset spectrograms against 11,853 events in the official e-CALLISTO catalog (2021–2024) reveals that 99.1% of training bursts are standard catalog events. In the clean test partition, however, **847 bursts (60.1%)** do not appear in the catalog under any station (and 66.5% do not match for that specific station); these were added during post-hoc manual re-inspection by the Principal Investigator. Equivalent faint features in training were labeled as background ($y=0$). Restricting evaluation strictly to standard catalog bursts contracts the observational recall gap by **27.34 pp** (from 45.28 pp down to 17.94 pp), and threshold-free rank discrimination is near-identical ($\text{AUROC} > 0.977$, gap $< 1\%$).
3. **Controlled Retraining ($N=3$ Seeds, 6 Models)**: Contrasting a temporal `purged` training arm against an exact-size `random_control` across 3 independent seeds under the authors' exact recipe causally proves that multi-station event leakage inflates recall, but its causal contribution is **modest: between 0 and 6 pp** depending on operating point (+5.78 pp [95% $t$-interval: 2.68, 8.87] at conservative calibration thresholds; +0.52 pp [95% $t$-interval: −4.52, 5.56] at test-matched operational false alarm rates). The majority of the apparent 45.28 pp deficit is explained by label protocol asymmetry and event detectability/size.
4. **Test-Set Optimization Forensics**: Hyperparameter optimization sweep configs (`configs/sweep_*.yml`) targeted `test_avg_f1` with `val_split: test`, serving as an interactive development scaffold that inadvertently compromised test-set sequestration.
5. **Operational Base-Rate Sensitivity**: At realistic space weather event prevalences ($\pi \in [0.001, 0.01]$), Bayes' theorem establishes that operational precision (PPV) drops to **7.9% – 46.5%**, yielding a 53% to 92% false alarm rate in continuous single-station deployment.

---

## Replication Pipeline (01 → 07)

The repository provides end-to-end reproducible scripts in [`analysis/`](analysis/):

| Step | Script | Description | Primary Output |
|:---:|:---|:---|:---|
| **01** | [`analysis/01_run_v2_inference.py`](analysis/01_run_v2_inference.py) | Downloads checkpoint `i4ds/flaresense-v2/model.ckpt` from Hugging Face and generates predictions on test set ($N=30,549$) | `results/core_results.json` |
| **02** | [`analysis/02_core_analysis.py`](analysis/02_core_analysis.py) | Computes micro/macro metrics, identifies $\pm 15$m / $\pm 60$m event leakage, and measures the 45.28 pp gap | `results/core_results.json` |
| **03** | [`analysis/03_breakdowns.py`](analysis/03_breakdowns.py) | Computes monotonic exposure-response gradient, station-wise degradation, and probability quantiles | `results/breakdown_results.json` |
| **04** | [`train/train_leakage_experiment.py`](train/train_leakage_experiment.py) | Controlled retraining framework: executes `purged` vs `random_control` across 3 random seeds ($N=3$, 6 models total) | `checkpoints/*.pt` |
| **05** | [`analysis/05_catalog_label_origin.py`](analysis/05_catalog_label_origin.py) | Cross-matches all splits against official e-CALLISTO catalog; identifies 847 uncatalogued clean bursts | `results/catalog_label_origin.json` |
| **06** | [`analysis/06_evaluate_retraining.py`](analysis/06_evaluate_retraining.py) | Evaluates retrained checkpoints on test split; computes double-difference contrasts ($\Delta\Delta$) | `results/retraining_experiment.json` |
| **07** | [`analysis/07_robustness_and_decomposition.py`](analysis/07_robustness_and_decomposition.py) | Evaluates multi-regime robustness (Calib-Fixed, Matched FPR, Matched Recall, AUROC) and two-step gap reduction | `results/decomposition_and_robustness.json` |

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/FlareSense-v2-Audit/FlareSense-v2-Audit.git
cd FlareSense-v2-Audit

# Install dependencies
pip install -r requirements.txt
```

### 2. Instant Offline Replication (Using Included Data)

All required evaluation artifacts, cached split metadata, and official e-CALLISTO catalog lists (~15 MB total) are included directly in [`data/`](data/) and [`results/`](results/). The entire audit pipeline (`02` through `07`) can be executed immediately and fully offline without GPU hardware or downloading the 63 GB spectrogram dataset:

```bash
# 1. Run core event-level leakage analysis
python analysis/02_core_analysis.py

# 2. Run exposure gradient and station breakdowns
python analysis/03_breakdowns.py

# 3. Verify catalog label origin & uncatalogued burst counts
python analysis/05_catalog_label_origin.py

# 4. Evaluate retrained models (purged vs random_control)
python analysis/06_evaluate_retraining.py

# 5. Evaluate multi-regime robustness & causal double-differences
python analysis/07_robustness_and_decomposition.py

# 6. Generate publication figures (Figures 0, 1, 2, 3, and 4)
python analysis/generate_concept_diagram.py
python analysis/generate_figures.py
python analysis/generate_comprehensive_figure.py
```

### 3. End-to-End Retraining (Optional, Requires GPU)

To retrain all 6 models from scratch under the authors' exact recipe:

```powershell
# Runs 3 seeds x 2 arms (purged vs random_control) on GPU
powershell -ExecutionPolicy Bypass -File train/run_overnight.ps1
```

---

## Two-Step Gap Reduction Summary

Rather than imposing an uncalibrated additive decomposition across mismatched thresholds, the 45.28 pp observational recall deficit is structured into two verifiable steps:

| Step / Dimension | Scope / Operating Regime | Metric / Effect | Interpretation / 95% Confidence Interval |
|---|---|:---:|---|
| **Step 1: Observational Scope**<br>*(Published FlareSense-v2)* | All Test Bursts ($n=2,689$)<br>Catalog Bursts ($n=1,842$) | 45.28 pp gap<br>17.94 pp gap | Raw published observational deficit<br>Standard e-CALLISTO catalog bursts |
| | **Scope Reduction** | **$-$27.34 pp** | **Label protocol asymmetry + event detectability / SNR** |
| **Step 2: Causal Leakage**<br>*(Retraining Contrast:)*<br>*Random Control − Purged*<br>*(3-Seed Pooled Mean)* | Calib-Fixed (Catalog)<br>Calib-Fixed (All Bursts)<br>Matched FPR$^*$ (Catalog)<br>Matched FPR$^*$ (All Bursts)<br>Matched Recall$^*$ (Catalog)<br>Matched Recall$^*$ (All Bursts)<br>Threshold-Free AUROC | **+5.78 pp**<br>**+4.09 pp**<br>**+0.52 pp**<br>**+2.21 pp**<br>**+1.68 pp**<br>**+2.86 pp**<br>**+0.001** | [2.68, 8.87] (positive across 3/3 seeds)<br>[2.75, 5.42] (positive across 3/3 seeds)<br>[$-$4.52, 5.56] (not significant; 2/3 seeds $>0$)<br>[$-$0.54, 4.95] (marginal; 3/3 seeds $>0$)<br>[$-$1.18, 4.53] (not significant; 3/3 seeds $>0$)<br>[2.06, 3.66] (positive across 3/3 seeds)<br>Rank difference $< 0.1\%$ in catalog |

$^*$*Matched FPR and Matched Recall are sensitivity analyses with thresholds aligned on the test split.*

---

## Publication Package & Figures

All publication figures and the complete LaTeX submission package are available in [`paper/`](paper/):

- [`paper/main.tex`](paper/main.tex): Full LaTeX manuscript with balanced environments and verified citations (release tag `v1.0.1`).
- [`paper/references.bib`](paper/references.bib): Bibliography containing verified Crossref DOIs and arXiv identifiers.
- [`paper/arxiv_submission.zip`](paper/arxiv_submission.zip): Self-contained upload bundle for Overleaf / arXiv.
- [`figures/`](figures/): Vector PDF and 300 DPI PNG figures:
  - `fig0_concept_leakage_mechanism.pdf`: Physical multi-station leakage diagram.
  - `fig1_leakage_disparity.pdf`: Observational recall disparity and confidence shift.
  - `fig2_exposure_and_did.pdf`: Monotonic exposure-response curve and difference-in-differences.
  - `fig3_station_degradation.pdf`: Instrument-level degradation across ground stations.
  - `fig4_causal_retraining_contrast.pdf`: Operating regime sensitivity and causal contrast forest plot ($N=3$).
- [`legacy/`](legacy/): Archived exploratory scripts superseded by the verified `analysis/` pipeline.

---

## Citation

```bibtex
@article{filosofov2026flaresense,
  title   = {Independent Evaluation Audit of {FlareSense}-v2:
             Multi-Station Event Leakage, Asymmetric Label Protocols,
             and Methodological Forensics in Solar Radio Burst Classification},
  author  = {Filosofov, M.},
  year    = {2026},
  url     = {https://github.com/Farrior13/FlareSense-v2-Audit}
}
```

---

## License

This replication package is licensed under the [MIT License](LICENSE).
