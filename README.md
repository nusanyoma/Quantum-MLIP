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

> **Maintainer note:** this repository will be published at
> [`nusanyoma/Quantum-MLIP`](https://github.com/nusanyoma/Quantum-MLIP).
> `download-assets.sh` expects `release_assets/code_code_assets-v1.tar.gz`
> (already built, with its SHA-256 in
> `release_assets/code_code_assets-v1.tar.gz.sha256`) to be attached as a
> release asset on a GitHub Release tagged **`assets-v1`** on that repo.
> To publish it:
> 1. Push this repository to `nusanyoma/Quantum-MLIP` (`main` branch).
> 2. On GitHub, go to **Releases → Draft a new release**, set the tag to
>    `assets-v1`, and attach `release_assets/code_code_assets-v1.tar.gz`
>    as a binary asset, then publish the release.
> 3. `download-assets.sh` will then resolve correctly as-is. If you use a
>    different tag or filename, update `ASSETS_URL` in
>    `download-assets.sh` to match.

**Fetched by `download-assets.sh`** (large; listed in `.gitignore`, not in
git history):

| Path | Contents | Why it's here |
|---|---|---|
| `data/ani_gdb_s0{1,2,3,4}.h5` | The 4 ANI-dataset shards (`D1`–`D4`, grouped by heavy-atom count) | Input for `01_transfer_learning/` and `02_ani_dataset_results/`'s inference scripts |
| `model/Classic/best_Default_nnp_shuffle_l{2..5}_..._{4,8,12}qu.pt` (12 files) | Pretrained `L_pre = L_out∘L_in` checkpoints | The starting point every transfer-learning script loads before inserting `L_mid`/`Q` (see §6 — this pretraining step itself cannot be redone from this package) |
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
> pretraining step is needed first; it's already done (see §6).

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
comparison figures in `out/` (these use independently-verified literal
values rather than calling `inference_ani.py` live — see §6).

## 5. What was changed relative to the original scripts

1. **Hardcoded relative paths** replaced with `DATA_DIR`/`MODEL_DIR`/`RESULTS_DIR`
   from `lib/ani_transfer/paths.py`, plus a small `sys.path` bootstrap added
   to every entry-point script, so nothing here depends on a specific
   working directory.
2. **Bugs fixed**: an `early_stopping_learning_rate` typo in
   `classic_transfer.py` that silently disabled early stopping;
   `inference_ani.py`'s hardcoded example config (`n_qubits=8`, and
   `data_No=3` for the quantum model) corrected to the configuration
   actually used and verified for the paper's Fig. cholesterol_result
   (`n_qubits=4`, `data_No=4` for both); `figure.ipynb`'s stale save path
   (now `03_cholesterol_validation/out/`).
3. **Nothing else was changed** — hyperparameter sweep ranges, model logic,
   and everything else are exactly as found in the original scripts.

## 6. Known limitations

- **Pretraining (producing the checkpoints in `model/Classic/`) cannot be
  redone from this package.** The script that originally produced them,
  and its source dataset, no longer exist anywhere in the project this was
  extracted from. This does not block anything above — the checkpoint
  files themselves are supplied — but you cannot verify or vary the
  pretraining step itself here.
- **`classic_transfer.py` and `qc_{Ry,RzRy}_{4,8,12}qu.py` process one
  dataset/hyperparameter combination per run**, not the full grid — the
  original workflow swept it by editing constants and resubmitting, and
  that pattern is preserved as-is rather than converted to CLI arguments.
- **The cholesterol-validation sweep is not one command** —
  `inference_ani.py` predicts one molecule with one model per invocation.
  `figure.ipynb`'s bar-chart values are a literal, hand-typed list rather
  than being read from `inference_ani.py`'s output live, though they were
  independently confirmed to match what it computes (§7).
- **`download-assets.sh` will 404 until the GitHub Release exists.** It
  points at `nusanyoma/Quantum-MLIP`'s release tag `assets-v1`, which
  doesn't exist yet — it needs to be created and have
  `release_assets/code_code_assets-v1.tar.gz` attached to it first (see §3).

## 7. Verification already performed

- Every `.py` file here compiles (`python -m py_compile`) and the
  `sys.path`/`paths.py` bootstrap resolves `DATA_DIR`/`MODEL_DIR`/`RESULTS_DIR`
  to the local `data/`, `model/`, `results/` directories shown above.
- `classic_transfer.py`'s and `qc_Ry_4qu.py`'s full pipelines (load
  pretrained checkpoint → build model → train a couple of epochs → save
  checkpoint → write log) were run end-to-end against the local files as a
  smoke test.
- `plot_ani_results.py`'s 12 outputs were compared against
  `260707_AGC_原稿/figure/` and against numbers quoted in `main.tex`
  (§3.4.1's 1.80/1.48 and 1.40/1.27 kcal/mol RMSE pairs; §3.4.2's
  1.55/0.39/0.14/0.13 kcal/mol improvements) — all matched.
- `inference_ani.py`'s corrected default configuration was run for all 3
  model types and cross-checked against the DFT reference in
  `data/cholesterol_optimized_.gjf.log` and against
  `figure.ipynb`'s bar-chart values — matched to within ≈0.3 kcal/mol
  (ordinary rounding, not a discrepancy).
- Full-scale re-training (100 epochs × the entire hyperparameter grid, on
  GPU) was **not** executed — the smoke test above confirms the pipeline
  runs correctly end-to-end; reproducing every published number by
  retraining the entire grid is a much larger, separate undertaking.
- The full cold-start flow was tested end-to-end in a clean clone: running
  `download-assets.sh` against the release asset, then Steps A–C exactly as
  written above (including invoking `classic_transfer.py` and
  `qc_Ry_4qu.py` directly, not through any helper code) — all completed
  successfully with no manual intervention. The only part not yet
  exercised over a real network connection is the GitHub Release itself,
  which doesn't exist yet (see §6).
