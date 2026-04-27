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
| adapter_bottleneck | 16 |
| adapter_location | residual |
| adapter_train_classifier | True |

## Model Parameter Metrics
| Metric | Value |
| --- | --- |
| total_params | 11246272 |
| num_trainable_params | 87040 |
| num_adapter_params | 81920 |
| trainable_param_ratio | 0.0077 |

## Final Metrics
| Metric | Value |
| --- | --- |
| final_avg_accuracy | 0.8119 |
| average_forgetting | 0.0198 |
| num_unlearning_events | 1 |

## Unlearning Metrics
| Metric | Value |
| --- | --- |
| Fu | -0.0048 |
| WorstDrop | 0.0005 |
| Au | 0.5000 |
| t_reset | 0.0059 |
| t_retrain | 0.0686 |
| t_forget_total | 3.2721 |
| num_updated_params | 69632 |
| s_share_ratio | 0.0000 |
| s_share_crit_ratio | 0.0000 |

## Overlap CSV Summary
| Metric | Value |
| --- | --- |
| overlap_csv | NA |

## Artifacts
| File | Location |
| --- | --- |
| config.json | runs/cifar10/T5_F3/pall_adapter/seed_0/20260426_175812/config.json |
| metrics.json | runs/cifar10/T5_F3/pall_adapter/seed_0/20260426_175812/metrics.json |
| results.pth | runs/cifar10/T5_F3/pall_adapter/seed_0/20260426_175812/results.pth |
| summary.txt | runs/cifar10/T5_F3/pall_adapter/seed_0/20260426_175812/summary.txt |
| overlap.csv | NA |
