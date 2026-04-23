# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar100 |
| method | pall_original |
| method_variant | original |
| seed | 0 |
| arch | subnet_resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| n_epochs | 1 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.3357 |
| average_forgetting | 0.1060 |
| num_unlearning_events | 3 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0200 |
| WorstDrop | 0.0760 |
| Au | 0.2000 |
| t_reset | 0.0014 |
| t_retrain | 0.1682 |
| t_forget_total | 0.1696 |
| num_updated_params | 1764117 |
| s_share_ratio | 0.7895 |
| s_share_crit_ratio | 1.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar100/T10_F3/pall_original/seed_0/20260423_155015/config.json |
| metrics.json | runs/cifar100/T10_F3/pall_original/seed_0/20260423_155015/metrics.json |
| results.pth | runs/cifar100/T10_F3/pall_original/seed_0/20260423_155015/results.pth |
| summary.txt | runs/cifar100/T10_F3/pall_original/seed_0/20260423_155015/summary.txt |
| overlap.csv | NA |
