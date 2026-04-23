# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar100 |
| method | pall_original |
| method_variant | original |
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
| final_avg_accuracy | 0.4226 |
| average_forgetting | 0.1350 |
| num_unlearning_events | 3 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0291 |
| WorstDrop | 0.0680 |
| Au | 0.2000 |
| t_reset | 0.0014 |
| t_retrain | 0.1759 |
| t_forget_total | 0.1773 |
| num_updated_params | 1764691 |
| s_share_ratio | 0.7898 |
| s_share_crit_ratio | 1.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar100/T10_F3/pall_original/seed_1/20260423_161436/config.json |
| metrics.json | runs/cifar100/T10_F3/pall_original/seed_1/20260423_161436/metrics.json |
| results.pth | runs/cifar100/T10_F3/pall_original/seed_1/20260423_161436/results.pth |
| summary.txt | runs/cifar100/T10_F3/pall_original/seed_1/20260423_161436/summary.txt |
| overlap.csv | NA |
