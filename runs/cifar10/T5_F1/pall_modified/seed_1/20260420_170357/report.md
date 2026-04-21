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
| n_forget | 1 |
| n_epochs | 3 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.9296 |
| average_forgetting | 0.0938 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0000 |
| WorstDrop | 0.0000 |
| Au | 0.4995 |
| t_reset | 0.0014 |
| t_retrain | 0.2647 |
| t_forget_total | 0.2661 |
| num_updated_params | 446641 |
| s_share_ratio | 0.2000 |
| s_share_crit_ratio | 0.2000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| n_tasks_in_overlap | 5 |
| num_task_pairs | 10 |
| avg_overlap_offdiag | 0.1111 |
| max_overlap_offdiag | 0.1114 |
| min_overlap_offdiag | 0.1109 |
| avg_overlap_all | 0.2889 |
| diag_mean | 1.0000 |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F1/pall_modified/seed_1/20260420_170357/config.json |
| metrics.json | runs/cifar10/T5_F1/pall_modified/seed_1/20260420_170357/metrics.json |
| results.pth | runs/cifar10/T5_F1/pall_modified/seed_1/20260420_170357/results.pth |
| summary.txt | runs/cifar10/T5_F1/pall_modified/seed_1/20260420_170357/summary.txt |
| overlap.csv | runs/cifar10/T5_F1/pall_modified/seed_1/20260420_170357/overlap.csv |
