# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar10 |
| method | pall_original |
| method_variant | original |
| seed | 1 |
| arch | subnet_resnet18 |
| class_per_task | 2 |
| n_tasks | 5 |
| n_forget | 1 |
| n_epochs | 3 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.9307 |
| average_forgetting | 0.0917 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0000 |
| WorstDrop | 0.0000 |
| Au | 0.4995 |
| t_reset | 0.0013 |
| t_retrain | 0.2266 |
| t_forget_total | 0.2279 |
| num_updated_params | 446641 |
| s_share_ratio | 0.2000 |
| s_share_crit_ratio | 1.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F1/pall_original/seed_1/20260420_165016/config.json |
| metrics.json | runs/cifar10/T5_F1/pall_original/seed_1/20260420_165016/metrics.json |
| results.pth | runs/cifar10/T5_F1/pall_original/seed_1/20260420_165016/results.pth |
| summary.txt | runs/cifar10/T5_F1/pall_original/seed_1/20260420_165016/summary.txt |
| overlap.csv | NA |
