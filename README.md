# Reproduction guide — "Quantum machine learning interatomic potential"

This directory is a reproduction package for the numerical results and
figures in the paper ["Quantum machine learning interatomic
potential"](https://arxiv.org/abs/2607.27841) (arXiv:2607.27841), covering
§3.3–3.5. Everything needed to run every script here — the pretrained
model checkpoints and a small slice of datasets/results — is provided,
but the large binary files (ANI datasets, model checkpoints, per-epoch
logs, result pickles) are **not committed to this repository**. They're
fetched on demand by `./download-assets.sh` into `data/`, `model/`, and
`results/`, to keep the git history small. See §3.

## 1. Requirements

Python 3.9, plus:

```
pip install -r requirements.txt
```

## 2. Directory structure

```
code/code/
├── README.md
├── requirements.txt
├── download-assets.sh                    fetches data/, model/, results/ (§3)
├── .gitignore
├── lib/ani_transfer/                    shared library — model definitions,
│                                         circuit ansätze, dataset loading,
│                                         and lib/ani_transfer/paths.py
│                                         (the DATA_DIR/MODEL_DIR/RESULTS_DIR
│                                         used by every script below)
├── data/                                 fetched by download-assets.sh (§3).
├── model/                                pretrained checkpoints, fetched by
│                                         download-assets.sh.
├── results/                              fetched + generated logs.
├── 01_transfer_learning/                 §3.3 — train the transfer models
│   ├── classic/classic_transfer.py       classical L_mid insertion
│   └── quantum/qc_{Ry,RzRy}_{4,8,12}qu.py  quantum-circuit Q insertion
├── 02_ani_dataset_results/               §3.4 — evaluate + plot
│   ├── cc_inference.py                   test-set eval (classical)
│   ├── qc_inference_{Ry,RzRy}.py         test-set eval (quantum)
│   ├── plot_ani_results.py               → figures in out/
│   └── compr_circuit_s01_CNOT.ipynb      original figure notebook (reference)
└── 03_cholesterol_validation/            §3.5 — practical-molecule validation
    ├── inference_ani.py                  per-molecule energy prediction
    ├── transform_coordinates.py          Gaussian16 .gjf.log → .xyz
    └── figure.ipynb                       → *_barplot.png in out/
```

## 3. Fetch the data and model checkpoints

```
./download-assets.sh
```

Run this once before anything else. `data/`, `model/`, and `results/` are
**not** part of this repository's git history (see `.gitignore`) — they're
~106MB of datasets and checkpoints, hosted externally and fetched/extracted
by this script instead.

**Fetched by `download-assets.sh`** (large; listed in `.gitignore`, not in
git history):

| Path | Contents | Why it's here |
|---|---|---|
| `data/ani_gdb_s0{1,2,3,4}.h5` | The 4 ANI-dataset shards (`D1`–`D4`, grouped by heavy-atom count) | Input for `01_transfer_learning/` and `02_ani_dataset_results/`'s inference scripts |
| `model/Classic/best_Default_nnp_shuffle_l{2..5}_..._{4,8,12}qu.pt` (12 files) | Pretrained `L_pre = L_out∘L_in` checkpoints | The starting point every transfer-learning script loads before inserting `L_mid`/`Q` (this pretraining step itself cannot be redone from this package) |
| `model/Classic_transfer/best_CC_s04_{1,2,3}layer_4qu_seed0.pt`, `model/Quantum_transfer/best_QC_s04_{1,2,3}layer_4qu_d1_seed0_all_RzRy.pt` | Already-trained transfer models for the cholesterol validation's exact configuration (`D4`, `n_q=4`, seed 0) | Lets `03_cholesterol_validation/inference_ani.py` run immediately |
| `results/valid/{Classic_transfer,Quantum_transfer}/*.txt` (39 files) | Per-epoch validation-RMSE logs for `D1` (`l=1..4`, `n_q=4`, seed 0, both ansätze, `d∈{1,5}`) and `D1`–`D4` (`l=1`, `n_q=4`, seeds 0–4) | Lets `plot_ani_results.py` draw the learning-curve panels and the `improvement_dataset.png` baseline immediately |
| `02_ani_dataset_results/*.pickle` | `CC_rmse_dict.pickle` / `QC_rmse_dict_{Ry,RzRy}.pickle` — aggregated test-RMSE over the full `(D, n_q, l, d, seed)` grid | Lets `plot_ani_results.py` draw the bar-chart figures immediately, without needing the full training grid re-run |

**Already in git** (small enough to commit directly):

| Path | Contents | Why it's here |
|---|---|---|
| `data/{cholesterol,cholestanol,epicholesterol,7-dehydrocholesterol}_optimized*.xyz` | DFT-optimized geometries | Input for `03_cholesterol_validation/inference_ani.py` |
| `data/*.gjf.log` | The Gaussian16 DFT logs the `.xyz` files were extracted from | Reference energies, for independently checking `inference_ani.py`'s output against DFT |

If you run the training scripts in §4, they will add further checkpoints
and logs to `model/` and `results/` alongside what `download-assets.sh`
fetched.

## 4. Step-by-step reproduction

### Step A — Transfer learning (§3.3)

```
cd 01_transfer_learning/classic
python classic_transfer.py
```

Trains the classical `L_mid`-inserted model, sweeping `n_qubits∈{4,8,12}`
and `l∈{1..4}` over 5 seeds, for **one** dataset at a time. Edit the
`for data_No in range(2, 3):` line in `main()` to pick `D1`–`D4`, and rerun
once per dataset to cover the full grid. Writes checkpoints to
`model/Classic_transfer/` and per-epoch logs to `results/{train,valid}/Classic_transfer/`.

```
cd 01_transfer_learning/quantum
python qc_Ry_4qu.py      # or qc_Ry_8qu.py / qc_Ry_12qu.py / qc_RzRy_*qu.py
```

Same idea for the quantum-circuit-inserted model, at the qubit count in the
filename. Edit `main()`'s loops to pick `l`, circuit depth `d∈{1,5}`, and
dataset. Writes to `model/Quantum_transfer/` and
`results/{train,valid}/Quantum_transfer/`.

> Both scripts load the pretrained checkpoint from `model/Classic/` — no
> pretraining step is needed first; it's already done.

### Step B — Aggregate results and draw the figures (§3.4)

```
cd 02_ani_dataset_results
python cc_inference.py            # only if you re-ran Step A and want fresh pickles
python qc_inference_Ry.py         # (same)
python qc_inference_RzRy.py       # (same)
python plot_ani_results.py
```

`plot_ani_results.py` is the one you actually need — it reads the
already-supplied `*_rmse_dict.pickle` files and the logs in `results/valid/`,
and writes `rmse_nq.png`, `rmse_l.png`, `rmse_d.png`, 8×
`learning_curve_l*_d*.png`, and `improvement_dataset.png` to `out/`. It
works immediately, without running Step A first. The `*_inference.py`
scripts are only needed if you reran Step A across the *entire*
hyperparameter grid and want to regenerate the pickles from your own
results instead of the supplied ones.

### Step C — Cholesterol validation (§3.5)

```
cd 03_cholesterol_validation
python inference_ani.py cholesterol_optimized_ original
python inference_ani.py cholesterol_optimized_ classic
python inference_ani.py cholesterol_optimized_ quantum
```

Each call predicts one molecule's energy with one model. Valid `file`
arguments (matching the `.xyz` files in `data/`):
`cholesterol_optimized_`, `cholestanol_optimized`, `epicholesterol_optimized`,
`7-dehydrocholesterol_optimized`. To compare against the DFT reference,
`grep "SCF Done" data/<file>.gjf.log | tail -1`.

Then open `figure.ipynb` and run all cells to produce the `*_barplot.png`
comparison figures in `out/` (these use precomputed literal values rather
than calling `inference_ani.py` live).
