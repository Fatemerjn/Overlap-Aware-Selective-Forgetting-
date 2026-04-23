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
| n_forget | 3 |
| n_epochs | 3 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.9034 |
| average_forgetting | 0.0936 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0016 |
| WorstDrop | 0.0060 |
| Au | 0.4960 |
| t_reset | 0.0019 |
| t_retrain | 0.2376 |
| t_forget_total | 0.2394 |
| num_updated_params | 1317615 |
| s_share_ratio | 0.5901 |
| s_share_crit_ratio | 1.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F3/pall_original/seed_1/20260423_134102/config.json |
| metrics.json | runs/cifar10/T5_F3/pall_original/seed_1/20260423_134102/metrics.json |
| results.pth | runs/cifar10/T5_F3/pall_original/seed_1/20260423_134102/results.pth |
| summary.txt | runs/cifar10/T5_F3/pall_original/seed_1/20260423_134102/summary.txt |
| overlap.csv | NA |
