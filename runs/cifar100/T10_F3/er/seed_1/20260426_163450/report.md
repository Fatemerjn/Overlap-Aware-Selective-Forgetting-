# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar100 |
| method | er |
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
| final_avg_accuracy | 0.5163 |
| average_forgetting | 0.1406 |
| num_unlearning_events | 3 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | 0.0003 |
| WorstDrop | 0.0300 |
| Au | 0.3380 |
| t_reset | NA |
| t_retrain | NA |
| t_forget_total | 5.4368 |
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
| config.json | runs/cifar100/T10_F3/er/seed_1/20260426_163450/config.json |
| metrics.json | runs/cifar100/T10_F3/er/seed_1/20260426_163450/metrics.json |
| results.pth | runs/cifar100/T10_F3/er/seed_1/20260426_163450/results.pth |
| summary.txt | runs/cifar100/T10_F3/er/seed_1/20260426_163450/summary.txt |
| overlap.csv | NA |
