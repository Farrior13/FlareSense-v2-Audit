# Supplementary: Repository Forensics — Raw Evidence

This document contains the raw forensic evidence supporting Section 6 of the main audit report. All data was extracted from the archived repository at [github.com/i4Ds/FlareSense-v2](https://github.com/i4Ds/FlareSense-v2) (commit `85c2f45`, main branch).

---

## A. Configuration Files

### A.1 sweep.yaml (complete)

```yaml
#https://docs.wandb.ai/guides/sweeps/quickstart
program: main.py
method: bayes
project: FlareSense-v2
entity: vincenzo-timmel
metric:
  name: test_avg_f1
  goal: maximize
parameters:
  general:
    parameters:
      max_epochs:
        max: 100
        min: 15
  model:
    parameters:
      lr:
        max: 1.0e-3
        min: 1.0e-6
      label_smoothing:
        max: 0.2
        min: 0.0
      model_type:
        values: [resnet34, resnet50, resnet101,resnet152]
      weight_decay:
        max: 1.0e-3
        min: 1.0e-9
      warmup_epochs:
        max: 20
        min: 3
  data:
    parameters:
      freq_mask_method:
        values: [mean, random, max, min]
      frequency_masking_para:
        max: 40
        min: 0
      time_masking_para:
        max: 90
        min: 0
      time_warp_w:
        max: 450
        min: 300

command:
  - ${interpreter}
  - ${program}
  - "--config"
  - "configs/test_v2.yml" #Insert your model here (do not push it, adapt it each time.)
```

### A.2 configs/best_v2.yml (complete)

```yaml
general:
  batch_size: 64
  max_epochs: 25
  binary: True
  use_random_sampler: False
  use_class_weights: True

model:
  model_type: resnet34
  input_size: [128, 512]
  optimizer_name: adamw
  warmup_epochs: 12
  lr: 0.00023762695665743765
  label_smoothing: 0.11739565855800864
  weight_decay: 0.0005164989055101516

data:
  train_path: [i4ds/ecallisto_radio_sunburst, i4ds/ecallisto_radio_sunburst]
  train_split: [train, val]
  val_path: i4ds/ecallisto_radio_sunburst
  val_split: test
  test_path: i4ds/ecallisto_radio_sunburst
  test_split: test
  train_label_name: manual_label
  val_label_name: manual_label
  test_label_name: manual_label
  antennas_train: []
  antennas_val: []
  antennas_test: []
  reduce_non_burst: False
  custom_resize: False
  clip_to_range: False
  train_class_to_reduce: 0
  reduction_fraction: 0.2
  use_augmentation: True
  frequency_masking_para: 25
  time_masking_para: 70
  freq_mask_method: random
  time_warp_w: 389
  pred_path: i4ds/ecallisto_radio_sunburst
  pred_split: test

# https://wandb.ai/vincenzo-timmel/FlareSense-v2/runs/dfpxq6wo/overview
```

### A.3 configs/test_v2.yml (complete)

```yaml
general:
  batch_size: 64
  max_epochs: 25
  binary: True
  use_random_sampler: False
  use_class_weights: True

model:
  model_type: resnet34
  input_size: [128, 512]
  optimizer_name: adamw
  warmup_epochs: 12
  lr: 0.00023762695665743765
  label_smoothing: 0.11739565855800864
  weight_decay: 0.0005164989055101516

data:
  train_path: [i4ds/ecallisto_radio_sunburst, i4ds/ecallisto_radio_sunburst]
  train_split: [train, val]
  val_path: i4ds/ecallisto_radio_sunburst
  val_split: test
  test_path: i4ds/ecallisto_radio_sunburst
  test_split: test
  train_label_name: manual_label
  val_label_name: manual_label
  test_label_name: manual_label
  antennas_train: []
  antennas_val: []
  antennas_test: []
  reduce_non_burst: False
  custom_resize: False
  clip_to_range: False
  train_class_to_reduce: 0
  reduction_fraction: 0.2
  use_augmentation: False
  frequency_masking_para: 0
  time_masking_para: 0
  freq_mask_method: random
  time_warp_w: 0
  pred_path: i4ds/ecallisto_radio_sunburst
  pred_split: train

# https://wandb.ai/vincenzo-timmel/FlareSense-v2/runs/dfpxq6wo/overview
```

---

## B. All val_split values across all configs

| Config file | `val_split` | Dataset | In main pipeline? |
|-------------|-----------|---------|-------------------|
| `configs/best_v2.yml` | **test** | `i4ds/ecallisto_radio_sunburst` | Yes (README, main.sh, sweep) |
| `configs/test_v2.yml` | **test** | `i4ds/ecallisto_radio_sunburst` | Yes (all 4 sweep configs) |
| `configs/relabeled_data.yml` | val | `i4ds/ecallisto_radio_sunburst` | No |
| `configs/relabeled_data_best.yml` | val | `i4ds/ecallisto_radio_sunburst` | No |
| `configs/relabel_test_only.yml` | validation | `radio-sunburst-ecallisto-paths-df-v2-TEST-SET-SPLIT` | No (old dataset) |
| `configs/pred.yml` | validation | `ecallisto_radio_sunburst-mai-october` | No (different dataset) |
| `configs/barlow_test.yml` | train | `radio-sunburst-ecallisto-paths-df-v2-TEST-SET-SPLIT` | No (Barlow Twins, old dataset) |

---

## C. Git commit history for configs/best_v2.yml

### C.1 Creation (2025-01-05)

```
commit 76892292b6cdbfaf4a181692ecbd1d874be05985
Author: Vincenzo Timmel <vincenzo.timmel@fhnw.ch>
Date:   Sun Jan 5 20:08:52 2025 +0100

    Adds best config run.

 configs/best_v2.yml | 41 +++++++++++++++++++++++++++++++++++++++++
 main.sh             |  9 ++++++---
 2 files changed, 47 insertions(+), 3 deletions(-)
```

File created with `val_split: test` and rounded hyperparameters (lr: 0.0002376).

### C.2 HP precision update (2025-01-06)

```
commit e17211deb7c1df1b0eb90e2792c7cb72308dfa78
Author: Vincenzo Timmel <vincenzo.timmel@fhnw.ch>
Date:   Mon Jan 6 01:12:36 2025 +0100

    Adds updated values for best model.

--- a/configs/best_v2.yml
+++ b/configs/best_v2.yml
-  lr: 0.0002376
-  label_smoothing: 0.117395659
-  weight_decay: 0.000516499
+  lr: 0.00023762695665743765
+  label_smoothing: 0.11739565855800864
+  weight_decay: 0.0005164989055101516
```

The HP values were updated from rounded to full-precision float values. This is consistent with copying exact values from a W&B sweep run log.

---

## D. Git commit history for configs/test_v2.yml

### D.1 Creation (2024-12-26)

```
commit 7c455bb003037799de1b6edc32ad12e03046efc2
Author: Vincenzo Timmel <vincenzo.timmel@fhnw.ch>
Date:   Thu Dec 26 16:13:56 2024 +0100

    Updated new method for resizing.
```

Created with `val_split: test`, `train_split: [train, val]`, `train_label_name: model_label`.

### D.2 Label name change (2024-12-27)

```
commit 792dbb0b0f4acea94853a475f12738416dbe5301
Date:   Fri Dec 27 11:20:17 2024 +0100

-  train_label_name: model_label
-  val_label_name: model_label
+  train_label_name: manual_label
+  val_label_name: manual_label
```

### D.3 time_masking_para update (2024-12-30)

```
commit 451b8a4b32c9f4c577aa388d156c7cd63c2bd3d1
Date:   Mon Dec 30 11:52:52 2024 +0100

-  time_masking_para: 33
+  time_masking_para: 155
```

### D.4 Optimizer and epochs update (2024-12-30)

```
commit b282df3ef0c2b266d298d7c86fafd3ead2a0b419
Date:   Mon Dec 30 12:58:34 2024 +0100

-  max_epochs: 20
+  max_epochs: 100

-  optimizer_name: adam
+  optimizer_name: adamw
+  warmup_epochs: 1

-  time_warp_w: 613
+  time_warp_w: 450
```

### D.5 Final sync with best_v2 (2025-10-27)

```
commit a7d646351bba3752804b15ce2718d04b9aea8b24
Date:   Mon Oct 27 19:23:27 2025 +0100

-  max_epochs: 100
+  max_epochs: 25

-  warmup_epochs: 1
-  lr: 0.0023439
-  label_smoothing: 0.12465
-  weight_decay: 8.9648e-7
+  warmup_epochs: 12
+  lr: 0.00023762695665743765
+  label_smoothing: 0.11739565855800864
+  weight_decay: 0.0005164989055101516

-  use_augmentation: True
-  frequency_masking_para: 77
-  time_masking_para: 155
-  freq_mask_method: min
-  time_warp_w: 450
+  use_augmentation: False
+  frequency_masking_para: 0
+  time_masking_para: 0
+  freq_mask_method: random
+  time_warp_w: 0
```

Note: `val_split: test` was **never modified** in any of these commits.

---

## E. Code flow: how val_split becomes the training validation set

```
main.py:57    config = yaml.safe_load(file)
main.py:74    config = wandb.config           # overwrite with W&B (for sweep)
main.py:86    ds_valid = load_dataset(         # LOAD VALIDATION DATA
main.py:87        config["data"]["val_path"],
main.py:88        split=config["data"]["val_split"]  # → "test"
main.py:89    )
main.py:150   ds_valid = EcallistoDatasetBinary(ds_valid, ...)
main.py:177   val_dataloader = DataLoader(ds_valid, ...)
main.py:210   trainer = Trainer(val_check_interval=1.0)  # validate every epoch
main.py:217   trainer.fit(model, train_dataloaders, val_dataloaders=val_dataloader)
              ↓
ecallisto_model.py:118  def validation_step(self, batch, batch_idx):
ecallisto_model.py:146  def on_validation_epoch_end(self):
ecallisto_model.py:164      avg_f1 = torch.mean(torch.tensor(antenna_f1_scores))
ecallisto_model.py:165      self.log("val_avg_f1", avg_f1, prog_bar=True)  # → W&B
              ↓
sweep.yaml:7    name: test_avg_f1              # W&B maximizes this
sweep.yaml:8    goal: maximize
```
