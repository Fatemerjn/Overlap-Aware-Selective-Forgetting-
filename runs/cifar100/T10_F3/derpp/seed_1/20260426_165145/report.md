# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar100 |
| method | derpp |
| method_variant | NA |
| seed | 1 |
| arch | resnet18 |
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
| final_avg_accuracy | 0.5483 |
| average_forgetting | 0.1088 |
| num_unlearning_events | 3 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | -0.0086 |
| WorstDrop | 0.0360 |
| Au | 0.4020 |
| t_reset | NA |
| t_retrain | NA |
| t_forget_total | 7.3656 |
| num_updated_params | NA |
| s_share_ratio | NA |
| s_share_crit_ratio | NA |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar100/T10_F3/derpp/seed_1/20260426_165145/config.json |
| metrics.json | runs/cifar100/T10_F3/derpp/seed_1/20260426_165145/metrics.json |
| results.pth | runs/cifar100/T10_F3/derpp/seed_1/20260426_165145/results.pth |
| summary.txt | runs/cifar100/T10_F3/derpp/seed_1/20260426_165145/summary.txt |
| overlap.csv | NA |
