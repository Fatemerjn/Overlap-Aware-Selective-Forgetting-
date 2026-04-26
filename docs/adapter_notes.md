# Adapter Prototype Notes

`pall_adapter` is a minimal selective-forgetting prototype with a frozen ResNet backbone and per-task bottleneck adapters.

Known limitation:

- `pall_adapter uses adapter reset prototype, not full PALL overlap mask yet.`

Smoke test:

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 1 --n_epochs 1 --seed 0 --method pall_adapter
```

Optional classifier training:

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 1 --n_epochs 1 --seed 0 --method pall_adapter --adapter_train_classifier
```
