# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar10 |
| method | pall_modified |
| method_variant | modified |
| seed | 1 |
| arch | subnet_resnet18 |
| class_per_task | 2 |
| n_tasks | 5 |
| n_forget | 3 |
| n_epochs | 3 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.9108 |
| average_forgetting | 0.0900 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | -0.0058 |
| WorstDrop | 0.0025 |
| Au | 0.4845 |
| t_reset | 0.0014 |
| t_retrain | 9.3989 |
| t_forget_total | 9.4003 |
| num_updated_params | 1317615 |
| s_share_ratio | 0.5901 |
| s_share_crit_ratio | 0.2000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| n_tasks_in_overlap | 5 |
| num_task_pairs | 10 |
| avg_overlap_offdiag | 0.1111 |
| max_overlap_offdiag | 0.1113 |
| min_overlap_offdiag | 0.1110 |
| avg_overlap_all | 0.2889 |
| diag_mean | 1.0000 |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F3/pall_modified/seed_1/20260423_135057/config.json |
| metrics.json | runs/cifar10/T5_F3/pall_modified/seed_1/20260423_135057/metrics.json |
| results.pth | runs/cifar10/T5_F3/pall_modified/seed_1/20260423_135057/results.pth |
| summary.txt | runs/cifar10/T5_F3/pall_modified/seed_1/20260423_135057/summary.txt |
| overlap.csv | runs/cifar10/T5_F3/pall_modified/seed_1/20260423_135057/overlap.csv |
