# Privacy-Aware Lifelong Learning

This repository contains the implementation used for task-wise continual learning and unlearning experiments, including PALL baselines and overlap-aware selective forgetting variants.

Paper:
"Privacy-Aware Lifelong Learning"\
<em>Ozan Özdenizci, Elmar Rueckert, Robert Legenstein</em>\
ICLR 2025\
https://openreview.net/forum?id=UstOpZCESc

## Current Implementation (Codebase Behavior)

Implemented methods (`--method`) are:
- `sequential`
- `ewc`
- `lwf`
- `er`
- `derpp`
- `lsf`
- `clpu`
- `pall_original`
- `pall_modified`

Notes:
- `--method pall` is normalized to `pall_modified`.
- The current pipeline processes a sequence of train/forget requests over task IDs.

## `pall_original` vs `pall_modified`

- `pall_original`: baseline PALL-style forgetting/retraining flow without overlap-aware protection regularization.
- `pall_modified`: overlap-aware selective forgetting variant that can protect critical shared parameters and optionally adapt retraining effort (`protect_ratio`/`protect_threshold`, `lambda_protect`, `retrain_steps`, `adaptive_retrain`, etc.).

## Supported Datasets

- `cifar10`
- `cifar100`
- `tinyimagenet`

Current `cifar100` constraint in code:
- `--class_per_task` must be `5`.
- `--n_tasks` must be in `[1, 20]`.
- Tasks use the fixed CIFAR-100 superclass grouping in the current implementation; each task contains one superclass with 5 original classes.

## Forgetting Setting (Important)

- The setup is task-ID-aware (task-incremental style evaluation with known task IDs).
- Forget requests are script-generated during each run from `n_tasks`, `n_forget`, and seed; the model does not infer what to forget on its own.
- In evaluation, the task ID is explicitly used to select the task slice/logits.

## Example Commands

One CIFAR-10 baseline run (`pall_original`):
```bash
python -u main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_original --seed 0 --deterministic
```

One CIFAR-10 modified run (`pall_modified`):
```bash
python -u main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_modified --seed 0 --deterministic \
  --protect_ratio 0.2 --lambda_protect 0.1 --retrain_steps 50 --dump_overlap
```

One CIFAR-100 baseline run (`pall_original`):
```bash
python -u main.py --dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 \
  --arch resnet18 --method pall_original --seed 0 --deterministic
```

One CIFAR-100 modified run (`pall_modified`):
```bash
python -u main.py --dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 \
  --arch resnet18 --method pall_modified --seed 0 --deterministic \
  --protect_ratio 0.2 --lambda_protect 0.1 --retrain_steps 50 --dump_overlap
```

Aggregate run outputs:
```bash
python tools/aggregate_results.py --root runs --out results_summary.csv
```

Generate comparison tables:
```bash
python tools/make_comparison_table.py \
  --in results_summary.csv \
  --out-csv comparison_table.csv \
  --out-md comparison_table.md
```

## Run Artifacts and Outputs

Run directory pattern:
`runs/<dataset>/T<n_tasks>_F<n_forget>/<method>/seed_<seed>/<timestamp>/`

Key files:
- `config.json`: full run configuration (CLI-resolved arguments and run metadata).
- `metrics.json`: structured metrics, normalized final results, per-unlearning events, and forgetting metrics.
- `results.pth`: serialized tensors/stats and request sequences for the run.
- `summary.txt`: compact plain-text summary of final run metrics.
- `report.md`: compact Markdown report with config, final metrics, unlearning metrics, overlap summary (if present), and artifact paths.
- `overlap.csv`: overlap matrix dump (written when `--dump_overlap` is enabled).

Other artifacts:
- `events.log`: chronological run log.
- `checkpoints/`: created for run organization (may remain empty depending on workflow).
- `debug/`: additional unlearning debug dumps when `--debug_unlearning` is enabled.

## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@inproceedings{ozdenizci2025pall,
  title={Privacy-aware lifelong learning},
  author={Ozan {\"O}zdenizci and Elmar Rueckert and Robert Legenstein},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=UstOpZCESc}
}
```

## Acknowledgments

This work was supported by the Graz Center for Machine Learning (GraML). This research was funded in whole or in part by the Austrian Science Fund (FWF) [10.55776/COE12].

Parts of this code repository is based on the following works:

* https://github.com/Cranial-XIX/Continual-Learning-Private-Unlearning
* https://github.com/ihaeyong/WSN
