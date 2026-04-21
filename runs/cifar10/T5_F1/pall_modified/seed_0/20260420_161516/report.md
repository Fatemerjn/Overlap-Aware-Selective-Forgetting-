# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar10 |
| method | pall_modified |
| method_variant | modified |
| seed | 0 |
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
| final_avg_accuracy | 0.9199 |
| average_forgetting | 0.0687 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | -0.0010 |
| WorstDrop | 0.0000 |
| Au | 0.5000 |
| t_reset | 0.0013 |
| t_retrain | 0.3739 |
| t_forget_total | 0.3753 |
| num_updated_params | 1317858 |
| s_share_ratio | 0.5902 |
| s_share_crit_ratio | 0.2000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| n_tasks_in_overlap | 5 |
| num_task_pairs | 10 |
| avg_overlap_offdiag | 0.1110 |
| max_overlap_offdiag | 0.1113 |
| min_overlap_offdiag | 0.1108 |
| avg_overlap_all | 0.2888 |
| diag_mean | 1.0000 |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F1/pall_modified/seed_0/20260420_161516/config.json |
| metrics.json | runs/cifar10/T5_F1/pall_modified/seed_0/20260420_161516/metrics.json |
| results.pth | runs/cifar10/T5_F1/pall_modified/seed_0/20260420_161516/results.pth |
| summary.txt | runs/cifar10/T5_F1/pall_modified/seed_0/20260420_161516/summary.txt |
| overlap.csv | runs/cifar10/T5_F1/pall_modified/seed_0/20260420_161516/overlap.csv |
