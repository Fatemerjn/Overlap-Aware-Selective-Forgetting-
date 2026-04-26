# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar100 |
| method | pall_modified |
| method_variant | modified |
| seed | 1 |
| arch | subnet_resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| n_epochs | 3 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.5323 |
| average_forgetting | 0.0862 |
| num_unlearning_events | 3 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | -0.0040 |
| WorstDrop | 0.0200 |
| Au | 0.1980 |
| t_reset | 0.0014 |
| t_retrain | 12.4880 |
| t_forget_total | 12.4894 |
| num_updated_params | 1764520 |
| s_share_ratio | 0.7897 |
| s_share_crit_ratio | 0.2000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| n_tasks_in_overlap | 10 |
| num_task_pairs | 45 |
| avg_overlap_offdiag | 0.1110 |
| max_overlap_offdiag | 0.1113 |
| min_overlap_offdiag | 0.1106 |
| avg_overlap_all | 0.1999 |
| diag_mean | 1.0000 |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar100/T10_F3/pall_modified/seed_1/20260426_160130/config.json |
| metrics.json | runs/cifar100/T10_F3/pall_modified/seed_1/20260426_160130/metrics.json |
| results.pth | runs/cifar100/T10_F3/pall_modified/seed_1/20260426_160130/results.pth |
| summary.txt | runs/cifar100/T10_F3/pall_modified/seed_1/20260426_160130/summary.txt |
| overlap.csv | runs/cifar100/T10_F3/pall_modified/seed_1/20260426_160130/overlap.csv |
