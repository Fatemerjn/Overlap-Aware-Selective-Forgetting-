# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar10 |
| method | pall_original |
| method_variant | original |
| seed | 0 |
| arch | subnet_resnet18 |
| class_per_task | 2 |
| n_tasks | 5 |
| n_forget | 1 |
| n_epochs | 1 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.8949 |
| average_forgetting | 0.0641 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0062 |
| WorstDrop | 0.0250 |
| Au | 0.4990 |
| t_reset | 0.0014 |
| t_retrain | 0.8898 |
| t_forget_total | 0.8911 |
| num_updated_params | 1317633 |
| s_share_ratio | 0.5901 |
| s_share_crit_ratio | 1.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F1/pall_original/seed_0/20260419_161721/config.json |
| metrics.json | runs/cifar10/T5_F1/pall_original/seed_0/20260419_161721/metrics.json |
| results.pth | runs/cifar10/T5_F1/pall_original/seed_0/20260419_161721/results.pth |
| summary.txt | runs/cifar10/T5_F1/pall_original/seed_0/20260419_161721/summary.txt |
| overlap.csv | NA |
