# Legacy Scripts (Superseded, Known Issues)

The scripts in this directory (`legacy/scripts/`) represent early exploratory stages of the audit and have been **superseded** by the verified, production-grade pipeline in [`analysis/`](../analysis/).

### Known Issues in Legacy Scripts:
- `generate_figures.py`: Used fixed hardcoded constants (e.g. reported precision 90.6% and static TPR/FPR estimates) rather than deriving figures dynamically from verified JSON outputs.
- `reproduce_all.py`: Inefficiently loaded and decoded all 63 GB of image spectrograms from Hugging Face rather than operating on lightweight cached metadata.
- `stage_b_physical_degradation.py`: Contained an inverted sorting bug in station delta metrics (affecting Alaska/Norway rankings).

**For reliable and verified replication, use the canonical pipeline in [`analysis/01_run_v2_inference.py`](../analysis/01_run_v2_inference.py) through [`analysis/07_robustness_and_decomposition.py`](../analysis/07_robustness_and_decomposition.py).**
