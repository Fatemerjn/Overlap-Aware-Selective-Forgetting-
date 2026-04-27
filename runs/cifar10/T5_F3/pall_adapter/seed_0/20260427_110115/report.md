# Run Report

## Config Summary
| Key | Value |
| --- | --- |
| dataset | cifar10 |
| method | pall_adapter |
| method_variant | adapter |
| seed | 0 |
| arch | adapter_resnet18 |
| class_per_task | 2 |
| n_tasks | 5 |
| n_forget | 3 |
| n_epochs | 3 |
| batch_size | 32 |
| optim | sgd |
| lr | 0.01 |
| deterministic | False |
| adapter_bottleneck | 32 |
| adapter_location | residual |
| adapter_train_classifier | True |

## Model Parameter Metrics
| Metric | Value |
| --- | --- |
| total_params | 11328192 |
| num_trainable_params | 168960 |
| num_adapter_params | 163840 |
| trainable_param_ratio | 0.0149 |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.8125 |
| average_forgetting | 0.0130 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | -0.0025 |
| WorstDrop | -0.0010 |
| Au | 0.5000 |
| t_reset | 0.0055 |
| t_retrain | 0.2101 |
| t_forget_total | 3.4092 |
| num_updated_params | 135168 |
| s_share_ratio | 0.0000 |
| s_share_crit_ratio | 0.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F3/pall_adapter/seed_0/20260427_110115/config.json |
| metrics.json | runs/cifar10/T5_F3/pall_adapter/seed_0/20260427_110115/metrics.json |
| results.pth | runs/cifar10/T5_F3/pall_adapter/seed_0/20260427_110115/results.pth |
| summary.txt | runs/cifar10/T5_F3/pall_adapter/seed_0/20260427_110115/summary.txt |
| overlap.csv | NA |
