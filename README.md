# PES2TS

PES2TS (Potential Energy Surface to Transition State) builds a traceable, leakage-audited data foundation on top of the public Reaction-QM dataset: it downloads and checksum-verifies the B3LYP-D3/TZVP artifacts, turns them into a transition-state-free reactant/product inventory, quarantines all ground-truth transition-state and IRC data behind a single allow-listed accessor, adopts the authors' own reaction-level train/valid/test split with an independent DRFP near-duplicate audit, and emits deterministic trial and stratified cohort manifests. This repository currently contains the project scaffold only (package, pinned requirements, configuration, CLI skeleton, and test layout); the pipeline stages themselves land in subsequent work items.

Environment setup (dedicated conda env; `pip` only, no `uv`):

```bash
conda create -n pes2ts python=3.12 -y
conda run -n pes2ts pip install -r requirements.txt
conda run -n pes2ts python bin/pes2ts g0 --help
```

Installation note: `drfp==0.3.7` transitively pulls `xgboost` (which on Linux additionally installs the ~305 MB `nvidia-nccl-cu13` wheel), `openpyxl`, and `pre-commit`, plus regular dependencies such as `scipy` and `tqdm`. These extras are accepted for reproducibility and are not used directly by the pipeline code.

Tests: `conda run -n pes2ts python -m pytest` runs the synthetic-fixture suite only; checks that need network access or the ~12 GB real download are marked `realdata` and excluded by default (see `pytest.ini`).
