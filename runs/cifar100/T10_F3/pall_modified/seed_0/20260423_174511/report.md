# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar100 |
| method | pall_modified |
| method_variant | modified |
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
| final_avg_accuracy | 0.4526 |
| average_forgetting | 0.0702 |
| num_unlearning_events | 3 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0011 |
| WorstDrop | 0.0220 |
| Au | 0.2000 |
| t_reset | 0.0014 |
| t_retrain | 9.1821 |
| t_forget_total | 9.1834 |
| num_updated_params | 1764268 |
| s_share_ratio | 0.7896 |
| s_share_crit_ratio | 0.2000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| n_tasks_in_overlap | 10 |
| num_task_pairs | 45 |
| avg_overlap_offdiag | 0.1109 |
| max_overlap_offdiag | 0.1113 |
| min_overlap_offdiag | 0.1107 |
| avg_overlap_all | 0.1998 |
| diag_mean | 1.0000 |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar100/T10_F3/pall_modified/seed_0/20260423_174511/config.json |
| metrics.json | runs/cifar100/T10_F3/pall_modified/seed_0/20260423_174511/metrics.json |
| results.pth | runs/cifar100/T10_F3/pall_modified/seed_0/20260423_174511/results.pth |
| summary.txt | runs/cifar100/T10_F3/pall_modified/seed_0/20260423_174511/summary.txt |
| overlap.csv | runs/cifar100/T10_F3/pall_modified/seed_0/20260423_174511/overlap.csv |
