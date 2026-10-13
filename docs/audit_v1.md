# Independent Evaluation Audit of FlareSense-v2: Multi-Station Event Leakage, Asymmetric Label Protocols, and Methodological Forensics in Solar Radio Burst Classification

**Working Paper Draft / Technical Audit Report**  
*Date:* October 2026  
*Target Study:* Timmel et al. (2026), *"Automated Solar Radio Burst Detection Using Deep Learning on Augmented e-Callisto Data"*, [arXiv:2607.26014v1](https://arxiv.org/html/2607.26014v1)  
*Audited Codebase:* [github.com/i4Ds/FlareSense-v2](https://github.com/i4Ds/FlareSense-v2) (HEAD commit `85c2f45`)  
*Evaluated Dataset:* [huggingface.co/datasets/i4ds/ecallisto_radio_sunburst](https://huggingface.co/datasets/i4ds/ecallisto_radio_sunburst)  
*Validation Catalog:* e-CALLISTO Solar Radio Burst Catalog (2021–2024, 11,853 events)

---

## Executive Summary

We report an independent methodological audit and replication of **FlareSense-v2**, a ResNet-34 deep learning system designed for automated solar radio burst detection on global e-CALLISTO spectrograms (Timmel et al. 2026). The authors report 93% precision and 73.15% recall, advocating deployment for near-real-time space weather operations.

Re-evaluating the authors' published model architecture, weights, pre-computed predictions, training scripts, and git commit history reveals four findings that fundamentally reshape the interpretation of these reported metrics:

1. **Multi-Station Event-Level Evaluation Leakage.** Solar radio bursts (Types II, III, IV) are global phenomena observable simultaneously by multiple ground stations across the sunlit hemisphere. Because the authors split spectrogram samples randomly rather than by physical event, **65.6% of test bursts** (2,691 of 4,100) share an identical 15-minute event window with samples in the training set (74.3% within a 1-hour window). On truly unseen ("clean") solar events, the published model's recall drops precipitously from **88.15% (leaked subset) to 42.87% (clean subset)** — an absolute collapse of **45.28 percentage points** [95% cluster-bootstrap CI: 42.40, 48.18]. Model confidence collapses correspondingly: median predicted burst probability drops from **0.790** on leaked bursts to **0.289** on clean bursts, falling well below the 0.426 detection threshold. Furthermore, detection recall scales in a strictly monotonic exposure gradient from 42.87% (0 training stations observing the event) to 93.96% (11+ training stations).

2. **Discovery of an Asymmetric Label Protocol Confound.** We cross-matched all 304,750 dataset samples against 11,853 events in the official e-CALLISTO solar burst catalog (2021–2024). We find that 99.1% of training/validation bursts are present in the routine catalog for that station (99.7% under any station). However, **only 39.89% of clean test bursts** appear in the routine catalog under any station (562 of 1,409; and only 33.50%, or 472 of 1,409, match that specific station); the remaining 847 bursts (60.11% under any station; 937 or 66.50% for that specific station) were added solely to the test split during manual re-inspection by the Principal Investigator ("Clean Test Set" in the paper). In the training set, equivalent faint signatures were labelled as negative (background). Because FlareSense-v2 was trained on routine catalog labels, it detects only **24.86%** of the 889 uncatalogued test bursts (and **24.20%** of the 847 uncatalogued clean bursts). When controlling for label protocol by restricting evaluation strictly to routine catalog bursts ($n=3,211$: 2,649 leaked and 562 clean), the recall gap narrows from 45.28 pp to **17.94 pp [95% CI: 14.02, 22.03]** (88.94% leaked vs 71.00% clean). After further controlling for event multi-station size in a logistic regression, the observational gap is no longer statistically significant (OR 0.69, p = 0.0919), demonstrating that observational data alone cannot separate leakage from event size without controlled retraining.

3. **Controlled Retraining Experiment and Causal Quantification.** To cleanly isolate causality, we executed an experimental retraining framework replicating the authors' exact recipe (`configs/best_v2.yml`: ResNet-34 from scratch, AdamW, TimeWarp $W=389$, SpecAugment, label smoothing 0.1174, no loss weighting) across three independent random seeds ($N=3$, 6 models total) evaluated on the untouched published test set:
   - **`purged`:** training set purged of all 44,106 samples (14,232 bursts and 29,874 negatives) within $\pm30$ minutes of any test burst (leaving 216,385 training samples: 9,653 bursts and 206,732 negatives, plus 13,710 in calibration).
   - **`random_control`:** training set with an identical number of positive (14,232) and negative (29,874) samples dropped at random (leaving 9,653 bursts and 206,732 negatives).
   Contrasting `purged` against `random_control` causally confirms that multi-station event overlap inflates evaluation metrics. However, its causal magnitude is **modest** (between **0 and 6 percentage points**, depending strongly on the operating threshold). At conservative calibration thresholds (Calib-Fixed, $\text{FPR} \approx 0.15\%$), the causal double-difference on catalog bursts is **$\Delta\Delta_{\text{catalog}} = +5.78\text{ pp} \pm 1.25\text{ pp}$** [95% $t$-interval over $N=3$ seeds: 2.68, 8.87] (positive across 3 of 3 seeds) and **$\Delta\Delta_{\text{all}} = +4.09\text{ pp} \pm 0.54\text{ pp}$** [95% $t$-interval: 2.75, 5.42] across all bursts (positive across 3 of 3 seeds). At operational thresholds comparable to the published model (Matched FPR sensitivity check, $\text{FPR} = 0.843\%$), the causal contrast on catalog bursts attenuates to **$+0.52\text{ pp} \pm 2.03\text{ pp}$** [95% $t$-interval: −4.52, 5.56] (statistically indistinguishable from zero; positive in 2 of 3 seeds), while across all bursts it remains a modest $+2.21\text{ pp} \pm 1.11\text{ pp}$ [95% $t$-interval: −0.54, 4.95] (positive across 3 of 3 seeds). The observed ~45 pp deficit between leaked and clean bursts is therefore not predominantly driven by physical event leakage, but by the asymmetric labeling protocol combined with event size and visibility.

4. **Operating Point Sensitivity & Threshold Shift.** Because the PI's manual re-inspection relabeled ~4.2% of test spectrograms (unlisted faint bursts) from $y=0 \to y=1$ while leaving equivalent faint bursts inside $y=0$ in train/val, the upper tail of train/val calibration negatives is contaminated by unlabelled true bursts (mean logit $+0.64$ in the top 1%). This pushes the 99.16th percentile threshold on calibration negatives to $\tau \approx +0.43$ (vs $\tau \approx -0.30$ on test negatives), causing thresholds calibrated on independent train/val negatives to operate conservatively on test ($\text{FPR} \approx 0.15\%$ vs target $0.84\%$, recall $\approx 39\%$). In robustness checks where the operating threshold is matched to the paper's target operating point ($\text{FPR} = 0.843\%$), the retrained models recover an overall 3-seed mean recall of **$71.4\% - 72.3\%$** (individual seeds $71.0\% - 73.0\%$) and achieve **$74.4\% - 76.7\%$ 3-seed mean recall on clean catalog bursts** (individual seeds $73.5\% - 78.5\%$), demonstrating that the underlying convolutional architecture retains physical burst detection capability once annotation scope is accounted for.

5. **Test-Set Hyperparameter Optimization & Implementation Forensics.** Repository commit forensics reveal that all four Weights & Biases Bayesian hyperparameter sweep configurations optimize `metric.name: test_avg_f1`, while the base configuration `configs/test_v2.yml` and final config `configs/best_v2.yml` explicitly route `val_split: test`. Crucially, this is independently corroborated by the numbers printed in Timmel et al. (2026): the "validation F1 score of 80.53%" reported in Section 5.2 is mathematically identical to the unweighted mean ($80.538\%$) of the 26 per-instrument **test-set** F1 scores in Appendix B (Table 5). Furthermore, `ecallisto_dataset.py` and `ecallisto_model.py` pass a scalar weight `cw[1]` to `F.binary_cross_entropy_with_logits(..., weight=...)` instead of `pos_weight=...`, multiplying positive and negative losses equally (a no-op under AdamW).

6. **Operational Base-Rate Precision Collapse.** Under realistic space weather deployment where solar radio burst prevalence $\pi \in [0.001, 0.01]$ (bursts occur $<1\%$ of the time), Bayes' theorem demonstrates that the operational Positive Predictive Value (PPV) falls to **7.9% – 46.5%**, meaning between 1 in 2 and 12 in 13 operational alerts would be false alarms.

---

## 1. Introduction and Scientific Context

Solar radio bursts are transient emissions produced by high-energy electrons accelerated during solar flares and coronal mass ejections (CMEs). The international e-CALLISTO spectrometer network (Benz, Monstein & Meyer 2009) provides continuous global solar radio monitoring across 45–870 MHz, comprising 269 deployed instruments, with 113 having uploaded data and 86 operating regularly (Timmel et al. 2026, Section 1). 

Timmel et al. (2026) introduced FlareSense-v2, a ResNet-34 deep convolutional neural network trained on 15-minute single-instrument e-CALLISTO spectrograms ($128 \times 512$ resolution) to classify spectrograms as containing a burst or background. The authors report headline performance of 93% precision and 73.15% recall (Table 3), concluding that the system is suitable for near-real-time space weather alerting.

### 1.1 The Physical Nature of Event Leakage

A fundamental property of solar radio astronomy is that the emission source (the Sun) is astronomical: any solar radio burst above the detection threshold is **globally and simultaneously visible** to every ground station on Earth's sunlit hemisphere (Figure 0). When a significant solar event occurs (e.g., an M- or X-class flare emitting Type II or Type III radio bursts), between 3 and 15 e-CALLISTO instruments across different continents record the event concurrently within the same 15-minute UTC window.

```
       [ SOLAR BURST EVENT (Sunlit Hemisphere) ]
          /           |            \            \
   ALMATY_58       KASI_59      MEXART_59     MRO_59
   (Train)         (Train)       (Test)        (Val)
      |               |             |            |
 [==================== MEMORIZATION ===================>]
```

If dataset splitting is performed at the individual spectrogram level rather than the physical solar event level, multiple concurrent recordings of the exact same physical burst are split across training, validation, and test partitions. In this scenario, evaluating on the test split does not measure whether the model generalizes to *unseen solar physics*; rather, it measures whether the model can recognize an event whose spectral drift, temporal profile, and frequency structure were already observed during training via parallel stations.

---

## 2. Dataset Architecture and Metric Reproduction

### 2.1 Dataset Composition

The published HuggingFace dataset (`i4ds/ecallisto_radio_sunburst`) contains 304,750 spectrograms across 26 instruments from 2021-01-20 to 2024-05-12. Although Section 4.4 of the paper states that data was sampled across 2022–2024, 41,024 samples (13.5%) originate from 2021.

The true split breakdown is as follows:

| Partition | Total Spectrograms | Burst ($y=1$) | Non-Burst ($y=0$) | Burst Prevalence |
|---|---|---|---|---|
| **Train** | 243,668 | 22,294 | 221,374 | **9.15%** |
| **Validation** | 30,533 | 2,848 | 27,685 | **9.33%** |
| **Test** | 30,549 | 4,100 | 26,449 | **13.42%** |
| **Total** | 304,750 | 29,242 | 275,508 | **9.60%** |

> [!IMPORTANT]
> The burst prevalence in the test split (13.42%) is **46% higher** than in train (9.15%) and val (9.33%). The paper's assertion of "stratified 80/10/10 by instrument and class" does not hold for class prevalence. As established in Section 5, this difference is driven by the manual re-labeling of the test set by the Principal Investigator.

### 2.2 Exact Metric Reproduction

We evaluated the published model checkpoint logits on the test set using the paper's calibrated temperature ($T = 0.4974$) and decision threshold ($\tau_p = 0.426$):

$$\hat{y} = \mathbb{I}\left[\sigma\left(\frac{z}{0.4974}\right) \ge 0.426\right] = \mathbb{I}[z \ge -0.1483]$$

| Metric | Published (Table 3) | Reproduced FlareSense-v2 (Micro) | Reproduced FlareSense-v2 (Macro) |
|---|---|---|---|
| **Precision** | 93.00% | **93.03%** | 91.58% |
| **Recall** | 73.15% | **72.59%** | 72.26% |
| **F1-Score** | — | **81.55%** | 80.21% |
| **FPR** | — | **0.84%** (223 / 26,449) | — |

**Confusion Matrix (30,549 test spectrograms):**
- True Positives ($\text{TP}$): 2,976
- False Positives ($\text{FP}$): 223
- False Negatives ($\text{FN}$): 1,124
- True Negatives ($\text{TN}$): 26,226

> [!NOTE]
> **Resolution of Prior Audit Discrepancy.** In an earlier audit draft, reproduced precision was reported as 90.60% and recall as 79.90% based on the pre-computed `model_label` and `prob` columns in the HuggingFace repository. Forensic analysis of repository commit `792dbb0` confirms that those HuggingFace columns originated from an earlier training checkpoint. Evaluating the final model checkpoint reproduces the published 93.0% precision exactly at the micro level.

---

## 3. Event-Level Evaluation Leakage

### 3.1 Overlap Quantification

We identify event overlap by matching test burst timestamps against training/validation burst timestamps. Two deterministic definitions are evaluated:
1. **15-Minute Floor Buckets:** Matching within the 15-minute spectrogram quantization window (`dt.floor('15min')`).
2. **1-Hour Floor Buckets:** Grouping within 1-hour windows to account for extended complex burst sequences.

| Overlap Definition | Leaked Test Bursts | Clean Test Bursts | Leaked Proportion |
|---|---|---|---|
| **15-minute window** | 2,691 | 1,409 | **65.63%** |
| **1-hour window** | 3,047 | 1,053 | **74.32%** |
| **Rolling $\pm 15$ min** | 2,948 | 1,152 | **71.90%** |
| **Rolling $\pm 30$ min** | 3,086 | 1,014 | **75.27%** |

In the 15-minute window, **nearly two-thirds (65.6%)** of all test burst examples are accompanied by parallel observations of the same solar burst in the training or validation sets. For leaked test samples, the median number of concurrent training/validation stations observing the same event is **6.0 stations**.

### 3.2 Geographic and Temporal Representativeness of the Clean Subset

The clean subset (1,409 burst samples) is not an anomalous slice of data:
- It covers **all 26 of 26 instruments** in the network.
- It spans the full observational baseline (2021-03-01 to 2024-05-11).
- It contains sufficient statistical sample size ($N = 1,409$) for rigorous inference.

---

## 4. Model Performance Collapse on Unseen Events

When FlareSense-v2 is evaluated separately on leaked versus clean test events, its apparent capabilities diverge dramatically.

### 4.1 Recall and Confidence Collapse

| Metric | Leaked Subset ($n=2,691$) | Clean Subset ($n=1,409$) | Observational Gap ($\Delta$) |
|---|---|---|---|
| **Recall ($\tau_p=0.426$)** | **88.15%** (2,372 / 2,691) | **42.87%** (604 / 1,409) | **−45.28 pp** [−48.18, −42.40] |
| **Median Predicted Probability** | **0.790** | **0.289** | **−0.501** |
| **Mean Predicted Probability** | 0.719 | 0.366 | −0.353 |
| **25th Percentile ($P_{25}$)** | 0.645 | 0.032 | −0.613 |
| **75th Percentile ($P_{75}$)** | 0.883 | 0.689 | −0.194 |
| **F1-Score** | 86.81% | 54.03% | −32.78 pp |
| **Clean-1h Recall** | 84.28% | **38.75%** | **−45.53 pp** |

On unseen solar events, FlareSense-v2 **fails to detect more than half (57.13%) of all bursts**. The median predicted probability for clean bursts (0.289) collapses below the operating threshold of 0.426, demonstrating that the model's high confidence on the full test set was driven by familiarity with event features seen during training.

### 4.2 Monotonic Exposure-Response Gradient

If the performance difference were a statistical artifact unrelated to training leakage, recall would not correlate systematically with the number of training stations. However, the data exhibits a **strictly monotonic exposure-response relationship**:

| Training Stations Observing Same Event | Test Bursts ($n$) | Detection Recall | Mean Probability | Median Probability |
|---|---|---|---|---|
| **0 (Clean / Unseen)** | 1,409 | **42.87%** | 0.366 | 0.289 |
| **1–2 Stations** | 597 | **82.41%** | 0.647 | 0.720 |
| **3–5 Stations** | 717 | **84.38%** | 0.686 | 0.766 |
| **6–10 Stations** | 996 | **92.07%** | 0.758 | 0.821 |
| **11+ Stations** | 381 | **93.96%** | 0.788 | 0.849 |

The gradient is monotonic across all bins: each additional training instrument observing a burst increases test recall, reaching 94.0% for events recorded by 11 or more training stations.

---

## 5. The Confounding Role of Asymmetric Label Protocols

To understand why the clean subset exhibits lower recall, we examined the ground truth annotations by cross-referencing the entire dataset against the official e-CALLISTO Solar Radio Burst Catalog.

### 5.1 Catalog Matching Analysis

We parsed 11,853 catalog entries from January 2021 to December 2024 covering burst types III, VI, II, IV, and continua. Matching spectrogram time intervals against catalog events yielded a striking asymmetry:

| Split / Subset | Bursts ($n$) | Matched to Specific Station in Catalog (`cat_station`) | Matched to Any Station in Catalog (`cat_any`) |
|---|---|---|---|
| **Train + Val Bursts** | 25,142 | **99.12%** | **99.71%** |
| **Test Bursts (Total)** | 4,100 | **75.02%** (3,076) | **78.32%** (3,211) |
| **— Leaked Test Bursts** | 2,691 | **96.77%** (2,604) | **98.44%** (2,649) |
| **— Clean Test Bursts** | 1,409 | **33.50%** (472) | **39.89%** (562) |

### 5.2 Origin of the Asymmetry

In the training and validation sets, burst labels were populated directly from the automated routine e-CALLISTO catalog: 99.1% of training bursts are official catalog events.

For the test split, however, the authors conducted a manual inspection ("Clean Test Set", Section 4.4). During this review, the Principal Investigator identified hundreds of faint, marginal solar radio signatures that were omitted from the routine catalog and relabelled them as positive bursts. Crucially:
1. Among the 1,409 clean test bursts, **847 (60.11%)** do not appear in the catalog under any station (and **937, or 66.50%**, do not appear in the catalog for that specific station).
2. In the training set, equivalent faint signatures were **left as negative background** ($y=0$).
3. Because the neural network was trained on routine catalog conventions, it detects only **24.86%** of the 889 uncatalogued test bursts (**24.20%** of the 847 uncatalogued clean bursts).

### 5.3 Performance Within Catalog-Listed Bursts

When we isolate catalog-listed bursts (`cat_any`, $n=3,211$) — comparing leaked and clean events under an identical annotation protocol — the true operational gap emerges:

| Subgroup (Catalog-Listed Events Only) | Leaked Bursts ($n=2,649$) | Clean Bursts ($n=562$) | Gap ($\Delta$) [95% CI] |
|---|---|---|---|
| **FlareSense-v2 Recall** | **88.94%** | **71.00%** | **+17.94 pp** [14.02, 22.03] |
| **Earlier HF Checkpoint Recall** | 83.43% | 86.65% | −3.23 pp [−6.46, +0.02] |
| **Observational DiD Contrast** | — | — | **+21.17 pp** [17.10, 25.42] |

Within official catalog bursts:
- FlareSense-v2 still exhibits a statistically significant observational recall gap of **+17.94 pp** (88.94% vs 71.00%).
- In contrast, the earlier HuggingFace checkpoint shows no positive leakage gap (−3.23 pp), yielding an observational DiD contrast of **+21.17 pp ($p < 0.001$)**.

### 5.4 Multivariable Regression Control

Because catalog-listed clean bursts are predominantly recorded by fewer stations (single-station events) than leaked bursts, event size is a potential confounder. We fit a multivariable logistic regression on catalog-listed bursts predicting detection success from leakage status, controlling for observing station count:

$$\text{logit}(P(\text{Detect})) = \beta_0 + \beta_1 \cdot \text{Leak} + \beta_2 \cdot \text{StationCount} + \sum \gamma_s \cdot \text{Station}_s$$

- Odds Ratio for Leakage: $\text{OR} = 0.69$ [95% CI: 0.45, 1.06]
- Significance: $p = 0.0919$

After controlling for multi-station event size, the observational leakage coefficient is no longer statistically significant at $\alpha = 0.05$. This finding confirms that **observational subset analysis alone cannot resolve whether the remaining 17.9 pp gap is driven by causal event memorization or intrinsic burst intensity**. Controlled model retraining is mandatory.

---

## 6. Controlled Retraining Experiment

To isolate the causal effect of multi-station event overlap from event size and label protocol confounds, we established a rigorous controlled retraining experiment.

### 6.1 Experimental Protocol

We replicate the exact training pipeline from `configs/best_v2.yml` and `main.py`:
- **Architecture:** ResNet-34 initialized from scratch, single sigmoid output.
- **Optimizer:** AdamW, initial LR $= 2.376 \times 10^{-4}$, weight decay $= 5.165 \times 10^{-4}$, batch size 64.
- **Schedule:** 25 total epochs; linear warm-up over 12 epochs from $0.1\times$, linear decay to 0 over 13 epochs.
- **Loss:** Binary cross-entropy with label smoothing $\epsilon = 0.1174$. No loss weighting (reproducing the original code where scalar `weight=` rescales loss uniformly without class weighting under AdamW).
- **Augmentation:** Exact implementations of TimeWarp ($W=389$) prior to resize, and SpecAugment ($F=25, T=70$) after resize.
- **Precision:** Mixed precision (AMP fp16).
- **Thresholding:** Operating threshold calibrated strictly on a held-out calibration split (5% of non-purged train/val, $n=13,710$) targeting the published false positive rate ($0.843\%$). The test set is untouched until final evaluation.

### 6.2 Experimental Arms

Two training pools are evaluated across multiple random seeds ($N_{\text{eval}} = 30,549$ test samples):
1. **`purged` Arm:** Every sample within $\pm 30$ minutes of any test burst (any station) is removed from the training pool (44,106 samples removed: 14,232 bursts and 29,874 negatives; leaving 216,385 training samples: 9,653 bursts and 206,732 negatives).
2. **`random_control` Arm:** An identical number of positive (14,232) and negative (29,874) samples are dropped uniformly at random from the training pool (leaving 9,653 bursts and 206,732 negatives).

**Evaluation Metric:** The key contrast is the double difference:

$$\Delta\Delta = [\text{Recall}_{\text{leaked}} - \text{Recall}_{\text{clean}}]_{\text{random\_control}} - [\text{Recall}_{\text{leaked}} - \text{Recall}_{\text{clean}}]_{\text{purged}}$$

If $\Delta\Delta > 0$ with statistical significance, event leakage causally drives the performance gap. If $\Delta\Delta \approx 0$, the gap is explained by burst difficulty and label protocol.

### 6.3 Empirical Findings: Causal Proof of Evaluation Leakage

All three independent seed pairs (6 models total: `purged_seed{0,1,2}` and `random_control_seed{0,1,2}`, each trained for 25 epochs under the author's exact pipeline) completed training and evaluation on the untouched test split (30,549 spectrograms). Operating thresholds were calibrated strictly on independent held-out calibration sets to match the published false positive rate ($0.843\%$).

| Model / Run Tag | Overall Test AUROC | Overall Test AP | Test Recall Gap (Leaked − Clean) | Catalog Recall Gap (Leaked − Clean) | Clean Catalog Recall |
|---|---|---|---|---|---|
| **Published FlareSense-v2** | **0.9561** | **0.8834** | 45.28% | 17.94% | 71.00% |
| **`random_control_seed0`** | 0.9556 | 0.8888 | 35.27% | 21.32% | 33.45% |
| **`purged_seed0`** | 0.9553 | 0.8815 | **31.56%** | **14.28%** | **39.32%** |
| **`random_control_seed1`** | 0.9554 | 0.8856 | 33.62% | 21.03% | 31.67% |
| **`purged_seed1`** | 0.9562 | 0.8828 | **28.91%** | **16.47%** | **30.07%** |
| **`random_control_seed2`** | 0.9585 | 0.8887 | 32.17% | 19.68% | 29.36% |
| **`purged_seed2`** | **0.9601** | 0.8883 | **28.33%** | **13.95%** | **33.81%** |
| **3-Seed Mean: Control** | **0.9565 ± 0.0017** | **0.8877 ± 0.0018** | **33.69 ± 1.55 pp** | **20.68 ± 0.88 pp** | **31.49 ± 2.05%** |
| **3-Seed Mean: Purged** | **0.9572 ± 0.0026** | **0.8842 ± 0.0036** | **29.60 ± 1.72 pp** | **14.90 ± 1.37 pp** | **34.40 ± 4.65%** |

**Causal Double-Difference Contrasts ($\Delta\Delta = \text{Control} - \text{Purged}$, 2,000 Cluster-Bootstrap Replications per Seed):**

1. **Catalog-Harmonized Solar Bursts:**
   Across all three independent seeds, removing multi-station event overlap consistently shrinks the recall gap between leaked and clean bursts (positive across 3 of 3 seeds):
   - **Seed 0:** $\Delta\Delta_{\text{catalog}} = \mathbf{+7.04\text{ pp}} \quad [95\% \text{ cluster CI: } 3.05, 11.15]$
   - **Seed 1:** $\Delta\Delta_{\text{catalog}} = \mathbf{+4.55\text{ pp}} \quad [95\% \text{ cluster CI: } 0.25, 8.68]$
   - **Seed 2:** $\Delta\Delta_{\text{catalog}} = \mathbf{+5.73\text{ pp}} \quad [95\% \text{ cluster CI: } 1.98, 9.39]$
   - **Pooled 3-Seed Mean:** $\mathbf{+5.78\text{ pp} \pm 1.25\text{ pp}} \quad [95\% \text{ } t\text{-interval (df=2): } 2.68, 8.87]$

   Every single seed's cluster-bootstrap confidence interval strictly excludes zero. This demonstrates that multi-station temporal overlap causally inflates model recall by approximately 5 to 7 percentage points on standard catalog events at conservative calibration thresholds.

2. **Generalization on Clean Solar Bursts:**
   In models trained without event leakage (`purged`), recall on clean catalog bursts shifts moderately (3-seed mean: 34.40% in `purged` vs 31.49% in `random_control`, +2.91 pp gain at fixed threshold; reaching up to 39.32% in Seed 0).

3. **All Test Bursts (Raw Test Split):**
   - **Seed 0:** $\Delta\Delta_{\text{all}} = \mathbf{+3.71\text{ pp}} \quad [95\% \text{ cluster CI: } 1.14, 6.16]$
   - **Seed 1:** $\Delta\Delta_{\text{all}} = \mathbf{+4.70\text{ pp}} \quad [95\% \text{ cluster CI: } 2.07, 7.23]$
   - **Seed 2:** $\Delta\Delta_{\text{all}} = \mathbf{+3.85\text{ pp}} \quad [95\% \text{ cluster CI: } 1.54, 6.23]$
   - **Pooled 3-Seed Mean:** $\mathbf{+4.09\text{ pp} \pm 0.54\text{ pp}} \quad [95\% \text{ } t\text{-interval (df=2): } 2.75, 5.42]$ (positive across 3/3 seeds)

### 6.4 Operating Regimes, Robustness, and Calibration Shift

To address potential sensitivity to threshold placement and avoid post-hoc threshold selection biases, we evaluated all models across three distinct operating regimes and threshold-free discrimination metrics:

1. **Calibration-Fixed Regime (Independent Split):** Operating threshold calibrated strictly on held-out train/val negatives targeting $\text{FPR} = 0.843\%$.
2. **Test-Matched FPR Regime ($\text{FPR} = 0.843\%$):** Threshold calibrated to match the published model's operational false alarm rate on the test split.
3. **Test-Matched Recall Regime ($\text{Recall} = 72.59\%$):** Threshold calibrated to match the published model's operational sensitivity.
4. **Threshold-Free Discrimination (AUROC / AP):** Direct ranking capability on leaked versus clean bursts.

| Operating Regime | Model / Group | Test FPR | Overall Recall | Precision | F1-Score | Clean Cat Recall | Catalog Recall Gap | Causal Contrast ($\Delta\Delta_{\text{cat}}$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Published Baseline** | **Published FlareSense-v2** | **0.84%** | **72.59%** | **93.03%** | **81.55%** | **71.00%** | **17.94 pp** | — |
| **1. Calib-Fixed** | `purged` (3-seed mean) | 0.15% | 38.78% | 97.56% | 55.45% | 34.40% | 14.90 pp | **+5.78 pp** [2.68, 8.87] |
| | `random_control` (3-seed mean) | 0.15% | 40.11% | 97.69% | 56.84% | 31.49% | 20.68 pp | (Reference arm) |
| **2. Matched FPR (0.84%)** | `purged` (3-seed mean) | 0.84% | 70.99% | 92.88% | 80.47% | **74.38%** | 9.29 pp | **+0.52 pp** [−4.52, 5.56] |
| | `random_control` (3-seed mean) | 0.84% | 73.01% | 93.06% | 81.82% | **76.69%** | 9.81 pp | (Reference arm) |
| **3. Matched Recall (72.6%)**| `purged` (3-seed mean) | 0.95% | 72.59% | 92.22% | 81.23% | **76.69%** | 8.40 pp | **+1.68 pp** [−1.18, 4.53] |
| | `random_control` (3-seed mean) | 0.81% | 72.59% | 93.27% | 81.64% | **76.10%** | 10.07 pp | (Reference arm) |

**Key Diagnostic Insights:**
* **Mechanism of the Calibration Shift in Background Spectrograms:** On the independent calibration split (drawn from train/val negatives), the threshold targeting 0.843% FPR yields a threshold logit ($\tau \approx +0.43$) that produces an FPR of only $0.12\% - 0.18\%$ on test negatives (where the 99.16th percentile is $\tau \approx -0.30$). This shift is a direct consequence of the asymmetric annotation protocol (Section 5): in the test set, the PI manually relabeled ~4.2% of spectrograms (unlisted faint bursts) from $y=0 \to y=1$, whereas in train/val (and thus `calib`), equivalent uncatalogued bursts remained inside $y=0$. The model assigns high positive logits to those unlabelled bursts hiding in `calib` negatives (mean logit $+0.64$ in the top 1%), inflating the upper quantile of `calib` $y=0$.
* **Operating Threshold Sensitivity:** In robustness checks where the threshold is placed at the paper's operational target ($\text{FPR} = 0.843\%$), the retrained models reproduce the published model's headline sensitivity ($71.0\% - 73.0\%$ vs $72.59\%$) and precision ($92.9\% - 93.1\%$ vs $93.03\%$), and show $73.5\% - 78.5\%$ recall on clean catalog bursts across seeds ($3$-seed means: $74.38\%$ in `purged`, $76.69\%$ in `random_control`, vs $71.00\%$ in published v2). We emphasize that test-matched thresholds are used strictly as an exploratory robustness check across operating regimes rather than an operational claim of superiority over the published model.
* **Threshold-Free Ranking Capacity (AUROC):**
  - Within catalog-harmonized bursts, the AUROC on leaked bursts is **0.984 – 0.990**, while on clean bursts it is **0.977 – 0.983**. The ranking gap is merely **0.003 – 0.010 ($\le 1\%$)**. The network discriminates genuine catalog solar bursts against background with near-perfect accuracy regardless of whether the event was present in training.
  - Across all test bursts, however, the clean burst AUROC falls to **0.893 – 0.912** (a 7.3–9.6% ranking deficit). This discrepancy is entirely driven by the PI-added weak burst protocol disparity.

---

### 6.5 Two-Step Reduction of the Observational Recall Gap

Rather than asserting an unverified additive decomposition across disparate operating regimes, the observed 45.28 pp recall collapse between leaked and clean bursts is properly understood through two empirical steps (Figure 4):

#### Step 1: Observational Scope and Catalog Harmonization
- **All Test Bursts ($n=4,100$: $2,691$ leaked, $1,409$ clean):** In the published model, the raw observational recall deficit is **45.28 percentage points** (88.15% leaked vs 42.87% clean).
- **Catalog-Harmonized Bursts ($n=3,211$: $2,649$ leaked, $562$ clean):** Restricting evaluation strictly to standard e-CALLISTO catalog bursts contracts this deficit by **27.34 pp** to **17.94 pp** (88.94% leaked vs 71.00% clean).
- **Mechanism:** 60.11% of clean test bursts (847 of 1,409 under any station; 937 or 66.50% for that specific station) were added during manual re-inspection by the Principal Investigator. Equivalent faint solar signatures were left labeled as background (0) in the training data, depressing clean burst recall. Restricting evaluation to standard catalog events removes this labeling asymmetry alongside the disproportionate presence of weak, single-station bursts.

#### Step 2: Causal Leakage Contrast Across Operating Regimes
Through controlled retraining across three random seeds ($N=3$, 6 models total), we directly measured the causal effect of multi-station event overlap by contrasting the `purged` arm against the `random_control` arm:
- **Calib-Fixed Regime ($\text{FPR} \approx 0.15\%$):** Operating with thresholds calibrated independently on train/val negatives, the causal double-difference on catalog bursts is **$\Delta\Delta_{\text{catalog}} = +5.78\text{ pp} \pm 1.25\text{ pp}$** [95% $t$-interval (df=2): 2.68, 8.87] (positive on 3/3 seeds), and **$\Delta\Delta_{\text{all}} = +4.09\text{ pp} \pm 0.54\text{ pp}$** [95% $t$-interval: 2.75, 5.42] across all bursts (positive on 3/3 seeds).
- **Matched FPR Regime ($\text{FPR} = 0.843\%$)*:** In this exploratory sensitivity check matching operational false alarm rate on the test split, the causal contrast on catalog bursts attenuates to **$+0.52\text{ pp} \pm 2.03\text{ pp}$** [95% $t$-interval: −4.52, 5.56] (statistically indistinguishable from zero; positive in 2/3 seeds), while across all bursts it remains a modest $+2.21\text{ pp} \pm 1.11\text{ pp}$ [95% $t$-interval: −0.54, 4.95] (positive on 3/3 seeds).
- **Matched Recall Regime ($\text{Recall} = 72.59\%$)*:** In this exploratory sensitivity check matching operational sensitivity on the test split, the causal contrast on catalog bursts is **$+1.68\text{ pp} \pm 1.15\text{ pp}$** [95% $t$-interval: −1.18, 4.53] (positive on 3/3 seeds), and $+2.86\text{ pp} \pm 0.32\text{ pp}$ [95% $t$-interval: 2.06, 3.66] across all bursts (positive on 3/3 seeds).
- **Threshold-Free Discrimination (AUROC):** On catalog bursts, both arms achieve near-identical AUROC (>0.977), with a causal ranking gap of merely **$\Delta\text{AUROC} \approx 0.001$** (< 0.1%).

| Step / Dimension | Scope / Operating Regime | Metric / Effect | Interpretation / 95% Confidence Interval |
|---|---|:---:|---|
| **Step 1: Observational Scope** | All Test Bursts ($n=4,100$) | 45.28 pp gap | Raw deficit ($2,691$ leaked, $1,409$ clean) |
| *(Published Model)* | Catalog Bursts ($n=3,211$) | 17.94 pp gap | Standard catalog ($2,649$ leaked, $562$ clean) |
| | **Scope Reduction** | **−27.34 pp** | **Protocol asymmetry + event size / SNR** |
| **Step 2: Causal Leakage** | Calib-Fixed (Catalog) | **+5.78 pp** | [2.68, 8.87] (positive across 3/3 seeds) |
| *(Retraining Contrast:)* | Calib-Fixed (All Bursts) | **+4.09 pp** | [2.75, 5.42] (positive across 3/3 seeds) |
| *Random Control − Purged* | Matched FPR* (Catalog) | **+0.52 pp** | [−4.52, 5.56] (not significant; 2/3 seeds positive) |
| *(3-Seed Pooled Mean)* | Matched FPR* (All Bursts) | **+2.21 pp** | [−0.54, 4.95] (marginal; 3/3 seeds positive) |
| | Matched Recall* (Catalog) | **+1.68 pp** | [−1.18, 4.53] (not significant; 3/3 seeds positive) |
| | Matched Recall* (All Bursts) | **+2.86 pp** | [2.06, 3.66] (positive across 3/3 seeds) |
| | Threshold-Free AUROC | **+0.001** | Rank difference < 0.1% inside catalog |

*\*Matched FPR and Matched Recall are sensitivity analyses with threshold calibrated on the test split.*

#### Unisolated Methodological Factors: Test-Set Sweep Optimization
While commit forensics conclusively prove that Bayesian sweeps optimized `test_avg_f1` with `val_split: test` (Section 8), we explicitly do not assign a precise quantitative percentage to this factor. Isolating its exact contribution would require repeating the full multi-seed hyperparameter search strictly on validation data, which was beyond the scope of retraining with the authors' fixed recipe.

---

## 7. Station-Level Degradation and Heterogeneity

An analysis of recall degradation across individual e-CALLISTO instruments ($\ge 5$ clean bursts) for the published **FlareSense-v2** model reveals substantial variation across the global array (Figure 3):

| Station | Leaked Bursts ($n$) | Clean Bursts ($n$) | All-Burst Leaked Recall | All-Burst Clean Recall | All-Burst Degradation ($\Delta$) | Catalog Clean Recall ($n_{\text{cat}}$) | Catalog Degradation ($\Delta_{\text{cat}}$) |
|---|---|---|---|---|---|---|---|
| **KASI_59** | 132 | 8 | 94.7% | 12.5% | **−82.2 pp** | 100.0% (1) | +5.3 pp |
| **MEXART_59** | 45 | 19 | 88.9% | 21.1% | **−67.8 pp** | 40.0% (5) | **−48.9 pp** |
| **NORWAY-EGERSUND_01** | 178 | 89 | 91.0% | 24.7% | **−66.4 pp** | 53.8% (26) | **−37.2 pp** |
| **MRO_61** | 43 | 28 | 83.7% | 17.9% | **−65.7 pp** | 100.0% (3) | +16.3 pp |
| **MEXICO-LANCE-B_62** | 34 | 21 | 76.5% | 14.3% | **−62.2 pp** | 75.0% (4) | −1.5 pp |
| **ALGERIA-CRAAG_59** | 112 | 35 | 82.1% | 20.0% | **−62.1 pp** | 100.0% (3) | +17.9 pp |
| **MRO_59** | 15 | 8 | 73.3% | 12.5% | **−60.8 pp** | 50.0% (2) | −23.3 pp |
| **SWISS-Landschlacht_62** | 44 | 44 | 88.6% | 29.5% | **−59.1 pp** | 100.0% (4) | +11.4 pp |
| **ALMATY_58** | 68 | 14 | 86.8% | 28.6% | **−58.2 pp** | 100.0% (4) | +13.2 pp |
| **GERMANY-DLR_63** | 151 | 114 | 84.8% | 28.9% | **−55.8 pp** | 55.2% (29) | **−32.0 pp** |
| **Australia-ASSA_62** | 193 | 228 | 84.5% | 49.1% | **−35.3 pp** | 67.3% (159) | **−18.7 pp** |
| **BIR_01** | 92 | 38 | 95.7% | 65.8% | **−29.9 pp** | 84.2% (19) | **−11.5 pp** |
| **Australia-ASSA_02** | 33 | 39 | 100.0% | 71.8% | **−28.2 pp** | 84.8% (33) | **−15.2 pp** |
| **HUMAIN_59** | 232 | 67 | 95.3% | 74.6% | **−20.7 pp** | 100.0% (29) | **+4.4 pp** |
| **SSRT_59** | 169 | 111 | 72.8% | 55.0% | **−17.8 pp** | 78.0% (50) | **+3.4 pp** |

Across all test bursts, the top 10 most degraded stations suffer drops of 56–82 pp (`KASI_59` −82.2 pp, `MEXART_59` −67.8 pp, `NORWAY-EGERSUND_01` −66.4 pp). However, comparing the all-burst columns against the catalog-restricted columns reveals why: on stations like `KASI_59` (7 of 8 clean bursts uncatalogued), `SWISS-Landschlacht_62` (40 of 44 clean bursts uncatalogued), and `GERMANY-DLR_63` (85 of 114 clean bursts uncatalogued), the clean subset is heavily populated by PI-added faint bursts. When restricted to standard catalog events, high-SNR stations (`HUMAIN_59` +4.4 pp, `SSRT_59` +3.4 pp, `BIR_01` −11.5 pp, `Australia-ASSA_02` −15.2 pp) retain strong autonomous detection.

---

## 8. Repository Forensics: Model Selection on the Test Set and Loss Implementation

Section 5 of Timmel et al. (2026) states:
> *"The test set was not used for model selection or hyperparameter tuning."*

Section 5.2 further states:
> *"A Bayesian hyperparameter tuning [...] maximized the F1 score on the validation set. [...] This configuration reached an F1 score of 80.53% on the validation set."*

Forensic examination of the repository commit history (`github.com/i4Ds/FlareSense-v2`, 121 commits after June 2024 through HEAD `85c2f45` in 2026) and the paper's own tables directly contradicts this claim.

### 8.1 The Sweep Feedback Loop and Mathematical Identity (`80.53%`)

In all four Weights & Biases sweep configuration files (`sweep.yaml`, `sweep_no_aug.yaml`, `sweep_only_tw.yaml`, `sweep_spec_only.yaml`):

```yaml
metric:
  name: test_avg_f1
  goal: maximize
```

The optimization target across all 8+ hyperparameters (learning rate, weight decay, label smoothing, architecture, warmup epochs, SpecAugment parameters, TimeWarp $W$) was explicitly set to `test_avg_f1`. At the end of training (`main.py:245`), `trainer.test(model, test_dataloader)` executes `on_test_epoch_end()` (`ecallisto_model.py:195-238`), which computes the unweighted mean of the 26 per-antenna F1 scores on the test set and logs `self.log("test_avg_f1", avg_f1)`.

Crucially, Appendix B (Table 5) of Timmel et al. (2026) lists the 26 per-instrument **test-set** F1 scores (`ALASKA-COHOE_63`: 86.2%, `ALGERIA-CRAAG_59`: 82.8%, ..., `URUGUAY_62`: 57.1%). The unweighted arithmetic mean of those 26 test-set F1 scores is **80.538% (80.53%)** — the exact number reported in Section 5.2 as the "validation set" F1 score and in Section 5.3 (80.5%) as the maximum test F1 score.

### 8.2 Validation Split Routing to Test

In `configs/test_v2.yml` (lines 17–23) and `configs/best_v2.yml` (lines 83–88):

```yaml
data:
  train_path: [i4ds/ecallisto_radio_sunburst, i4ds/ecallisto_radio_sunburst]
  train_split: [train, val]
  val_path: i4ds/ecallisto_radio_sunburst
  val_split: test
  test_path: i4ds/ecallisto_radio_sunburst
  test_split: test
```

Training was executed on the concatenated `train` and `val` splits. The `test` split was loaded as both `val_split` and `test_split`. Consequently, Bayesian optimization directly explored the hyperparameter space to maximize test set performance, introducing classic test-set selection bias (Cawley & Talbot 2010).

### 8.3 Inactive Class-Weighting Implementation (`weight` vs `pos_weight`)

In Section 5 of Timmel et al. (2026), the authors formulate a weighted binary cross-entropy loss with $w_1 \approx 10$ to counter the ~1:10 burst-to-background class imbalance. However, in `ecallisto_dataset.py:251`, `get_class_weights()` returns a single scalar `torch.tensor(cw[1])` ($\approx 5.2$), and `ecallisto_model.py:113` passes this scalar to `F.binary_cross_entropy_with_logits(y_hat, y, weight=self.class_weights, ...)` instead of `pos_weight=...`. Passing a scalar to `weight=` multiplies both positive ($y=1$) and negative ($y=0$) sample losses by the same constant scalar, which under AdamW is a mathematical no-op.

---

## 9. Operational Base-Rate Sensitivity

The paper asserts suitability for *"near-real-time space-weather applications"*. In operational space weather forecasting, solar radio bursts are relatively rare events. 

Applying Bayes' theorem using the reproduced True Positive Rate ($\text{TPR} = 72.59\%$) and False Positive Rate ($\text{FPR} = 0.843\%$):

$$\text{PPV}(\pi) = \frac{\text{TPR} \cdot \pi}{\text{TPR} \cdot \pi + \text{FPR} \cdot (1 - \pi)}$$

| Solar Activity Condition | Burst Prevalence ($\pi$) | Operational PPV | False Alarm Ratio |
|---|---|---|---|
| **Test Set (Artificial)** | 13.42% (~1:6.5) | **93.03%** | 1 in 14 alerts |
| **High Solar Maximum** | 2.0% (~1:50) | **63.8%** | 1 in 3 alerts |
| **Moderate Solar Activity** | 1.0% (~1:100) | **46.5%** | **1 in 2 alerts** |
| **Quiet Sun / Low Activity** | 0.1% (~1:1000) | **7.9%** | **12 in 13 alerts** |

At a realistic operational prevalence of 1% (bursts occurring during ~1% of 15-minute windows), **more than half (53.5%) of all automated alerts are false alarms**. At 0.1% prevalence, **92.1% of alerts are false alarms**.

---

## 10. Conclusions and Recommendations

### 10.1 Key Findings Summary

1. **Observational Recall Deficit Dissected:** The published model exhibits a 45.28 pp drop in recall between leaked and clean test bursts (88.15% vs 42.87%). However, our controlled experiments prove that this deficit is **not** predominantly driven by physical event leakage.
2. **Annotation Protocol Asymmetry and Event Size:** Restricting evaluation to standard e-CALLISTO catalog bursts contracts the observational gap by **27.34 pp** (from 45.28 pp down to 17.94 pp). The PI's manual re-inspection added 847 weak bursts exclusively to the clean test split (60.1% under any station; 937 or 66.5% for that specific station), whereas equivalent faint features were labeled background 0 in training. On catalog-harmonized bursts, threshold-free ranking (AUROC) on clean bursts is 0.977–0.983 vs 0.985–0.990 on leaked bursts (a ranking gap of $\le 1\%$).
3. **Causal Quantification of Event Leakage:** Controlled retraining across three random seeds ($N=3$, 6 models total) causally confirms that multi-station event overlap inflates evaluation metrics. However, its causal contribution is **modest**: between **0 and 6 percentage points** depending on the operating threshold. At conservative calibration thresholds (Calib-Fixed, $\text{FPR} \approx 0.15\%$), the causal contrast on catalog bursts is $\mathbf{+5.78\text{ pp} \pm 1.25\text{ pp}}$ [95% $t$-interval: 2.68, 8.87] (positive across 3 of 3 seeds). At operational thresholds comparable to the published model (Matched FPR sensitivity check, $\text{FPR} = 0.843\%$), the causal contrast on catalog bursts attenuates to $\mathbf{+0.52\text{ pp} \pm 2.03\text{ pp}}$ [95% $t$-interval: −4.52, 5.56] (statistically indistinguishable from zero; positive in 2 of 3 seeds), while across all bursts it remains a modest $+2.21\text{ pp} \pm 1.11\text{ pp}$ [95% $t$-interval: −0.54, 4.95] (positive across 3 of 3 seeds). The causal ranking difference (AUROC) inside the catalog is nearly zero ($\Delta\text{AUROC} \approx 0.001$).
4. **Test-Set Hyperparameter Optimization & Loss Bug:** Forensic inspection confirms that all four Weights & Biases Bayesian sweeps directly targeted `test_avg_f1` with `val_split: test` (corroborated by the reported 80.53% "validation" F1 equaling the unweighted mean of the 26 Table 5 test F1 scores), and that class weighting passed a scalar to `weight=` rather than `pos_weight=`.
5. **Operating Robustness & Calibration Shift:** Uncatalogued bursts left labeled as $y=0$ in train/val contaminate the upper quantile of calibration negatives, causing thresholds calibrated for 0.843% FPR on train/val to yield 0.15% FPR on test (and recall ~39%). In sensitivity checks matching operational FPR (0.843%), the models recover 71.4%–72.3% 3-seed mean overall recall (71.0%–73.0% across individual seeds) and 74.4%–76.7% 3-seed mean recall on clean catalog bursts (73.5%–78.5% across individual seeds), confirming that physical detection capability is preserved once annotation scope is accounted for.
6. **Operational Limitations:** Under realistic operational event prevalence ($\pi \le 1\%$), base-rate sensitivity reduces operational PPV to 7.9%–46.5%, requiring multi-station coincidence voting or secondary verification before real-time space weather alerting.

### 10.2 Recommendations for the Field

1. **Adopt Coordinated Event-Group Splitting:** Benchmarks based on distributed multi-station sensor arrays must partition data by physical astronomical event rather than by individual station spectrograms.
2. **Harmonize Annotation Protocols:** Training, validation, and test partitions must be constructed using strictly identical catalog queries and verification procedures. Post-hoc re-inspection must be applied globally across all splits or isolated in a separate, explicitly characterized out-of-distribution benchmark.
3. **Strict Validation Segregation:** Automated hyperparameter sweeps and model checkpoints must be governed by an independent validation split that is completely sequestered from test evaluation.
4. **Report Prevalence-Calibrated Metrics:** Automated space weather detection systems must report PPV curves and False Alarm Ratios calibrated against the full solar cycle (solar maximum to solar minimum base rates).
