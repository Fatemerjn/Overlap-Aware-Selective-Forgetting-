# Privacy-Aware Lifelong Learning

This is the code repository of the following [paper](https://openreview.net/pdf?id=UstOpZCESc) on task-incremental lifelong learning and unlearning by leveraging task-specific sparse subnetworks with knowledge transfer.

"Privacy-Aware Lifelong Learning"\
<em>Ozan Özdenizci, Elmar Rueckert, Robert Legenstein</em>\
International Conference on Learning Representations (ICLR), 2025.\
https://openreview.net/forum?id=UstOpZCESc

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

## Reproducible Runs & Unlearning Diagnostics

Runs now write all artifacts to:
`runs/<dataset>/T<n_tasks>_F<n_forget>/<method>/seed_<seed>/<timestamp>/`

Saved files include:
- `config.json` (full run configuration)
- `metrics.json` (per-unlearning diagnostics and sanity checks)
- `events.log` (human-readable log)
- `summary.txt` (key numbers)
- `checkpoints/` (optional, empty by default)
- `overlap.csv` (if `--dump_overlap` is set)
- `debug/` (if `--debug_unlearning` is set)

Reproducibility:
- `--seed <int>` and `--deterministic` enforce seeded, deterministic behavior.

Baseline vs modified PALL:
- `--method pall_original` (baseline reset behavior)
- `--method pall_modified` (protected shared params; default for `--method pall`)

Protection knobs:
- `--protect_ratio` or `--protect_threshold` (critical shared params)
- `--lambda_protect` (regularization on protected params)
- `--retrain_steps` / `--retrain_epochs` and `--adaptive_retrain`
- `--debug_unlearning` and `--dump_overlap`

Example (baseline):
```bash
python -u main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_original --seed 0 --deterministic
```

Example (modified):
```bash
python -u main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_modified --seed 0 --deterministic \
  --protect_ratio 0.2 --lambda_protect 0.1 --retrain_steps 50 \
  --debug_unlearning --dump_overlap
```
