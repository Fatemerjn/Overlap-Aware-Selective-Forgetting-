import argparse
import csv
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from methods import *
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Privacy-Aware Lifelong Learning')
parser.add_argument('--data_dir', default='./data', type=str, help='data directory')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'])
parser.add_argument('--class_per_task', default=2, type=int, help='number of classes per task in CL')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks in CL')
parser.add_argument('--n_forget', default=3, type=int, help='number of forget requests by the user to simulate')
parser.add_argument('--arch', default='resnet18', type=str, help='neural network architecture')
parser.add_argument('--norm_params', default=False, action='store_true', help='use batch-norm params in dense models')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--gpu', default='0', type=str, help='CUDA device id')
parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                    help='compute device preference')

parser.add_argument('--n_epochs', default=20, type=int, help='number of iterations per task')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer choice')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')

parser.add_argument(
    '--method',
    default='pall',
    choices=[
        'sequential',
        'ewc',
        'lwf',
        'er',
        'derpp',
        'lsf',
        'clpu',
        'pall',
        'pall_original',
        'pall_modified',
    ],
    help='method for CL with unlearning',
)
parser.add_argument('--sparsity', default=0.8, type=float, help="layer-wise sparsity for PALL")
parser.add_argument('--mem_budget', default=500, type=int, help='rehearsal memory capacity')
parser.add_argument('--mem_type', default='random', choices=['random'])
parser.add_argument('--ewc_lmbd', default=100., type=float, help='EWC lambda parameter')
parser.add_argument('--lsf_gamma', default=10.0, type=float, help='LSF gamma parameter')
parser.add_argument('--lwf_alpha', default=1.0, type=float, help='LWF alpha parameter')
parser.add_argument('--lwf_temp', default=2.0, type=float, help='LWF temp parameter')
parser.add_argument('--alpha', default=0.5, type=float, help='DERPP alpha parameter')
parser.add_argument('--beta', default=1.0, type=float, help='DERPP beta parameter')
parser.add_argument('--k_shot', default=1, type=int, help='k-shot finetuning for PALL')
parser.add_argument('--forget_iters', default=None, type=int, help='forgetting iterations for ER methods')
parser.add_argument('--deterministic', default=False, action='store_true', help='enable deterministic runs')
parser.add_argument('--protect_ratio', default=None, type=float, help='fraction of shared params to protect')
parser.add_argument('--protect_threshold', default=None, type=float, help='abs weight threshold for protection')
parser.add_argument('--lambda_protect', default=0.0, type=float, help='regularization weight for protected params')
parser.add_argument('--retrain_steps', default=None, type=int, help='override retrain steps for PALL unlearning')
parser.add_argument('--retrain_epochs', default=None, type=int, help='alias for retrain steps (PALL)')
parser.add_argument('--allow_zero_retrain', default=False, action='store_true',
                    help='allow retrain_steps=0 to skip finetune without fallback')
parser.add_argument('--adaptive_retrain', default=False, action='store_true', help='adapt retrain steps to overlap')
parser.add_argument('--debug_unlearning', default=False, action='store_true', help='dump unlearning artifacts')
parser.add_argument('--dump_overlap', default=False, action='store_true', help='dump overlap matrix CSV')
args = parser.parse_args()

PALL_METHODS = {"pall", "pall_original", "pall_modified"}


def normalize_method(arg_namespace):
    if arg_namespace.method == "pall":
        arg_namespace.method = "pall_modified"
    if arg_namespace.method == "pall_modified":
        arg_namespace.method_variant = "modified"
    elif arg_namespace.method == "pall_original":
        arg_namespace.method_variant = "original"
    else:
        arg_namespace.method_variant = None


def set_seed(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def init_run_dir(arg_namespace):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_seq = f"T{arg_namespace.n_tasks}_F{arg_namespace.n_forget}"
    run_dir = Path("runs") / arg_namespace.dataset / task_seq / arg_namespace.method
    run_dir = run_dir / f"seed_{arg_namespace.seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir, timestamp


def init_logger(run_dir):
    logger = logging.getLogger("pall")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "events.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_event(logger, message):
    print(message)
    if logger is not None:
        logger.info(message)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def serialize_config(arg_namespace, run_dir, timestamp):
    config = vars(arg_namespace).copy()
    config["device"] = str(arg_namespace.device)
    config["run_dir"] = str(run_dir)
    config["timestamp"] = timestamp
    return config


normalize_method(args)


def resolve_device(arg_namespace):
    """Select an execution device with macOS MPS support."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if arg_namespace.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = arg_namespace.gpu
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but no CUDA devices are available.")

    if arg_namespace.device == "mps":
        if mps_available:
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available on this system.")

    if arg_namespace.device == "cpu":
        return torch.device("cpu")

    # Auto mode: prefer CUDA, then MPS, then CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = arg_namespace.gpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


args.device = resolve_device(args)
args.arch = 'subnet_' + args.arch.lower() if args.method in PALL_METHODS else args.arch.lower()
args.dim_input = (3, 64, 64) if args.dataset == "tinyimagenet" else (3, 32, 32)


def evaluate(test_datasets, args, model, return_logits=True, verbose=True):
    model.eval_mode()
    L, A = torch.zeros(args.n_tasks), torch.zeros(args.n_tasks)
    logits = [] if return_logits else None
    cpt = args.class_per_task
    with torch.no_grad():
        for task, dataset in enumerate(test_datasets):
            bsize = args.batch_size
            loader = DataLoader(dataset, batch_size=bsize, shuffle=False)
            l = a = n = 0.0
            logit_ = torch.zeros(len(dataset), cpt) if return_logits else None
            for i, (x, y) in enumerate(loader):
                x_tensor, y_tensor = x.to(args.device), y.to(args.device)
                y_ = model.evaluate(x_tensor, task)
                l += F.cross_entropy(y_, y_tensor, reduction='sum').item()
                a += y_.argmax(-1).eq(y_tensor).float().sum().item()
                if return_logits:
                    logit_[i * bsize:i * bsize + y_tensor.shape[0]].copy_(
                        y_[..., cpt * task:cpt * (task + 1)].cpu()
                    )
                n += y_tensor.shape[0]

            L[task], A[task] = l / n, a / n
            if return_logits:
                logits.append(logit_)

    model.train_mode()
    if verbose:
        print("[INFO] loss: ", L)
        print("[INFO] acc.: ", A)

    return {
        'loss': L,
        'accuracy': A,
        'logits': logits,
    }


def to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def avg_for_tasks(acc_list, task_ids):
    if not task_ids:
        return 0.0
    return float(sum(acc_list[t] for t in task_ids) / len(task_ids))


def acc_list_to_dict(acc_list):
    return {str(idx): float(acc) for idx, acc in enumerate(acc_list)}


def process_requests(args, model, train_datasets, test_datasets, requests, run_context):
    forgotten_tasks = []
    loss = torch.zeros(len(requests), args.n_tasks)
    accuracy = torch.zeros(len(requests), args.n_tasks)
    times = torch.zeros(len(requests))
    forgotten_tasks_mask = torch.zeros(len(requests), args.n_tasks)
    active_tasks_mask = torch.zeros(len(requests), args.n_tasks)
    logits = [torch.zeros(len(requests), len(ds), args.class_per_task) for ds in test_datasets]

    logger = run_context.get("logger")
    metrics_state = run_context.get("metrics_state")
    metrics_path = run_context.get("metrics_path")
    debug_dir = run_context.get("debug_dir")

    unlearning_step = 0

    for request_id, (task_id, learn_type, active_tasks) in enumerate(requests):
        log_event(logger, "============================================================")
        learn_type_str = {"T": "Training", "F": "Forgetting"}[learn_type]
        log_event(logger, f"[INFO] {learn_type_str} Task {task_id} ...")

        if learn_type == "F":
            forgotten_tasks.append(task_id)

        if learn_type == "F":
            pre_eval = evaluate(test_datasets, args, model, return_logits=False, verbose=False)
            pre_acc = to_list(pre_eval["accuracy"])
            remaining_tasks = list(active_tasks)
            avg_before = avg_for_tasks(pre_acc, remaining_tasks)

            def eval_callback(stage):
                return evaluate(test_datasets, args, model, return_logits=False, verbose=False)

            if hasattr(model, "forget_with_diagnostics"):
                if task_id not in model.task_status:
                    raise AssertionError(f"[ERROR] {task_id} was not learned")
                model.task_status[task_id] = "F"
                debug_context = None
                if args.debug_unlearning:
                    debug_context = {
                        "debug_dir": str(debug_dir),
                        "request_id": request_id,
                        "task_id": task_id,
                        "unlearning_step": unlearning_step,
                    }
                info = model.forget_with_diagnostics(
                    task_id,
                    eval_fn=eval_callback,
                    debug_context=debug_context,
                    remaining_tasks=remaining_tasks,
                )
            else:
                t0 = time.perf_counter()
                model.privacy_aware_lifelong_learning(task_id, train_datasets[task_id], learn_type)
                t1 = time.perf_counter()
                info = {
                    "t_reset": t1 - t0,
                    "t_retrain": 0.0,
                    "num_updated_params": 0,
                }

            after_reset_eval = info.get("after_reset_eval")
            after_reset_acc = to_list(after_reset_eval["accuracy"]) if after_reset_eval else pre_acc

            stat = evaluate(test_datasets, args, model, return_logits=True, verbose=True)
            post_acc = to_list(stat["accuracy"])

            avg_after_reset = avg_for_tasks(after_reset_acc, remaining_tasks)
            avg_after_retrain = avg_for_tasks(post_acc, remaining_tasks)
            fu = avg_before - avg_after_retrain
            worst_drop = 0.0
            if remaining_tasks:
                worst_drop = max(pre_acc[t] - post_acc[t] for t in remaining_tasks)
            au = post_acc[task_id] if task_id < len(post_acc) else 0.0
            finetune_diag = json_safe(info.get("finetune_diag", None))

            event = {
                "unlearning_step": unlearning_step,
                "request_id": request_id,
                "task_id": task_id,
                "remaining_tasks": remaining_tasks,
                "per_task_acc_before": acc_list_to_dict(pre_acc),
                "per_task_acc_after_reset": acc_list_to_dict(after_reset_acc),
                "per_task_acc_after_retrain": acc_list_to_dict(post_acc),
                "avg_before": avg_before,
                "avg_after_reset": avg_after_reset,
                "avg_after_retrain": avg_after_retrain,
                "Fu": fu,
                "WorstDrop": worst_drop,
                "Au": au,
                "t_reset": info.get("t_reset", 0.0),
                "t_retrain": info.get("t_retrain", 0.0),
                "num_updated_params": info.get("num_updated_params", 0),
                "overlap": {
                    "s_t": info.get("s_t", 0),
                    "s_share": info.get("s_share", 0),
                    "s_share_crit": info.get("s_share_crit", 0),
                    "s_share_ratio": info.get("s_share_ratio", 0.0),
                    "s_share_crit_ratio": info.get("s_share_crit_ratio", 0.0),
                },
                "protection": info.get("protection", {}),
                "finetune_diag": finetune_diag,
            }
            log_event(
                logger,
                "[INFO] overlap: |S_t|={s_t} |S_share|={s_share} |S_share_crit|={s_share_crit} "
                "ratios: share={share_ratio:.4f} crit={crit_ratio:.4f}".format(
                    s_t=info.get("s_t", 0),
                    s_share=info.get("s_share", 0),
                    s_share_crit=info.get("s_share_crit", 0),
                    share_ratio=info.get("s_share_ratio", 0.0),
                    crit_ratio=info.get("s_share_crit_ratio", 0.0),
                ),
            )
            log_event(
                logger,
                "[INFO] unlearning timing: t_reset={t_reset:.4f}s t_retrain={t_retrain:.4f}s "
                "updated_params={updated}".format(
                    t_reset=info.get("t_reset", 0.0),
                    t_retrain=info.get("t_retrain", 0.0),
                    updated=info.get("num_updated_params", 0),
                ),
            )
            if finetune_diag is not None:
                log_event(logger, f"[INFO] finetune_diag: {json.dumps(finetune_diag)}")
            if metrics_state is not None:
                metrics_state.setdefault("unlearning_events", []).append(event)
                write_json(metrics_path, metrics_state)

            chance_acc = 1.0 / args.class_per_task
            sanity_msgs = []
            if pre_acc[task_id] - au < 0.05:
                sanity_msgs.append(
                    f"[WARN] Unlearned task {task_id} accuracy drop is small: "
                    f"{pre_acc[task_id]:.4f} -> {au:.4f}"
                )
            if au > chance_acc + 0.05:
                sanity_msgs.append(
                    f"[WARN] Unlearned task {task_id} accuracy above chance: {au:.4f} (chance {chance_acc:.4f})"
                )
            share_ratio = info.get("s_share_ratio", 1.0)
            if args.method_variant == "modified" and share_ratio < 0.05:
                sanity_msgs.append(
                    f"[INFO] Overlap ratio is low ({share_ratio:.4f}); modified method should be close to baseline."
                )
            if sanity_msgs and metrics_state is not None:
                metrics_state.setdefault("sanity_checks", []).extend(sanity_msgs)
                write_json(metrics_path, metrics_state)
            for msg in sanity_msgs:
                log_event(logger, msg)

            unlearning_step += 1
            times[request_id] = info.get("t_reset", 0.0) + info.get("t_retrain", 0.0)
        else:
            t0 = time.perf_counter()
            model.privacy_aware_lifelong_learning(task_id, train_datasets[task_id], learn_type)
            t1 = time.perf_counter()
            stat = evaluate(test_datasets, args, model, return_logits=True, verbose=True)
            times[request_id] = t1 - t0

        # evaluate bookkeeping
        for forget_task in forgotten_tasks:
            forgotten_tasks_mask[request_id][forget_task] = 1.0
        for active_task in active_tasks:
            active_tasks_mask[request_id][active_task] = 1.0

        loss[request_id] = stat["loss"]
        accuracy[request_id] = stat["accuracy"]
        if stat["logits"] is not None:
            for t in range(args.n_tasks):
                logits[t][request_id] = stat["logits"][t]

    return {
        "loss": loss,
        "accuracy": accuracy,
        "times": times,
        "forgotten_tasks_mask": forgotten_tasks_mask,
        "active_tasks_mask": active_tasks_mask,
        "logits": logits,
    }


def write_overlap_csv(path, task_ids, matrix):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id"] + [str(t) for t in task_ids])
        for task_id, row in zip(task_ids, matrix):
            writer.writerow([str(task_id)] + [f"{val:.6f}" for val in row])


def write_summary(path, summary, unlearning_events):
    lines = [
        f"run_dir: {summary['run_dir']}",
        f"dataset: {summary['dataset']}",
        f"method: {summary['method']} ({summary['method_variant']})",
        f"seed: {summary['seed']} (deterministic={summary['deterministic']})",
        f"tasks: {summary['n_tasks']} | forget_requests: {summary['n_forget']}",
        f"final_avg_accuracy: {summary['final_avg_accuracy']:.4f}",
        "",
        "unlearning_events:",
    ]
    if not unlearning_events:
        lines.append("none")
    for event in unlearning_events:
        overlap = event.get("overlap", {})
        lines.append(
            "step {step} task {task} avg_before {avg_before:.4f} "
            "avg_after_reset {avg_after_reset:.4f} avg_after_retrain {avg_after_retrain:.4f} "
            "Fu {fu:.4f} WorstDrop {worst_drop:.4f} Au {au:.4f} "
            "t_reset {t_reset:.2f}s t_retrain {t_retrain:.2f}s "
            "share_ratio {share_ratio:.4f} crit_ratio {crit_ratio:.4f}".format(
                step=event.get("unlearning_step"),
                task=event.get("task_id"),
                avg_before=event.get("avg_before", 0.0),
                avg_after_reset=event.get("avg_after_reset", 0.0),
                avg_after_retrain=event.get("avg_after_retrain", 0.0),
                fu=event.get("Fu", 0.0),
                worst_drop=event.get("WorstDrop", 0.0),
                au=event.get("Au", 0.0),
                t_reset=event.get("t_reset", 0.0),
                t_retrain=event.get("t_retrain", 0.0),
                share_ratio=overlap.get("s_share_ratio", 0.0),
                crit_ratio=overlap.get("s_share_crit_ratio", 0.0),
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def generate_user_requests(num_tasks, sequence_length):
    if sequence_length < num_tasks:
        raise ValueError("Sequence length must be at least the number of tasks.")

    user_requests = [(i, "T") for i in range(num_tasks)]
    trained_tasks = list(range(num_tasks))

    remaining_slots = sequence_length - num_tasks
    f_requests = []
    while remaining_slots > 0 and trained_tasks:
        task = random.choice(trained_tasks)
        f_requests.append((task, "F"))
        trained_tasks.pop(trained_tasks.index(task))
        remaining_slots -= 1

    for f_request in f_requests:
        t_index = user_requests.index((f_request[0], "T"))
        valid_positions = list(range(t_index + 1, len(user_requests) + 1))
        insert_position = random.choice(valid_positions)
        user_requests.insert(insert_position, f_request)

    return user_requests


def get_request_datasets():
    def clear_all_forget_requests(li):
        to_be_removed = []
        for request_id, (task_id, learn_type, active_tasks) in enumerate(li):
            if learn_type == "F":
                to_be_removed.append(request_id)
                for j in range(request_id):
                    if li[j][0] == task_id and li[j][1] == "T":
                        to_be_removed.append(j)
                        break
        new_list, new_active_tasks = [], []
        for request_id, (task_id, learn_type, active_tasks) in enumerate(li):
            if request_id not in to_be_removed:
                if learn_type == "T" and (task_id not in new_active_tasks):
                    new_active_tasks.append(task_id)
                new_list.append((task_id, learn_type, list(new_active_tasks)))
        return new_list

    # Loading the datasets
    train_datasets, test_datasets = get_task_datasets(args)

    user_requests = generate_user_requests(num_tasks=args.n_tasks, sequence_length=int(args.n_tasks + args.n_forget))

    user_requests_with_active_tasks, active_tasks = [], []
    for task_id, learn_type in user_requests:
        if learn_type == "T" and (task_id not in active_tasks):
            active_tasks.append(task_id)
        elif learn_type == "F":
            active_tasks.remove(task_id)
        user_requests_with_active_tasks.append((task_id, learn_type, list(active_tasks)))
    print('user_requests_with_active_tasks: ', user_requests_with_active_tasks)

    user_requests_without_forgotten = []
    for request_id, (task_id, learn_type, active_tasks) in enumerate(user_requests_with_active_tasks):
        if learn_type == "F":
            list_up_to = list(user_requests_with_active_tasks[:request_id + 1])
            user_requests_without_forgotten.append(clear_all_forget_requests(list_up_to))
    print('user_requests_without_forgotten: ', user_requests_without_forgotten)

    return train_datasets, test_datasets, user_requests_with_active_tasks, user_requests_without_forgotten


def main():
    global args
    set_seed(args.seed, args.deterministic)

    run_dir, timestamp = init_run_dir(args)
    args.run_dir = str(run_dir)
    logger = init_logger(run_dir)
    config = serialize_config(args, run_dir, timestamp)
    write_json(run_dir / "config.json", config)

    methods_dict = {
        "sequential": Sequential,
        "ewc": EWC,
        "lwf": LwF,
        "er": ER,
        "derpp": Derpp,
        "lsf": LSF,
        "clpu": CLPU,
        "pall": PALL,
        "pall_original": PALL,
        "pall_modified": PALL,
    }

    log_event(logger, "============================================================")
    log_event(logger, "[INFO] -- Experiment Configs --")
    log_event(logger, "       1. data & task")
    log_event(logger, f"          dataset:      {args.dataset}")
    log_event(logger, f"          n_tasks:      {args.n_tasks}")
    log_event(logger, f"          # class/task: {args.class_per_task}")
    log_event(logger, "       2. training")
    log_event(logger, f"          lr:           {args.lr:5.4f}")
    log_event(logger, "       3. model")
    log_event(logger, f"          method:       {args.method} ({args.method_variant})")
    log_event(logger, f"          architecture: {args.arch}")
    log_event(logger, f"          norm params:  {args.norm_params}")
    log_event(logger, f"          device:       {args.device}")
    log_event(logger, f"          deterministic:{args.deterministic}")
    log_event(logger, f"          run_dir:      {run_dir}")
    log_event(logger, "============================================================")

    train_datasets, test_datasets, user_requests_with_active_tasks, user_requests_without_forgotten = get_request_datasets()
    log_event(logger, "[INFO] finish processing data")

    metrics_state = {
        "run": {
            "dataset": args.dataset,
            "method": args.method,
            "method_variant": args.method_variant,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "n_tasks": args.n_tasks,
            "n_forget": args.n_forget,
            "timestamp": timestamp,
            "run_dir": str(run_dir),
        },
        "unlearning_events": [],
        "sanity_checks": [],
        "requests": user_requests_with_active_tasks,
        "requests_without_forgotten": user_requests_without_forgotten,
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics_state)

    debug_dir = run_dir / "debug" if args.debug_unlearning else None
    if debug_dir is not None:
        debug_dir.mkdir(exist_ok=True)

    run_context = {
        "run_dir": run_dir,
        "logger": logger,
        "metrics_state": metrics_state,
        "metrics_path": metrics_path,
        "debug_dir": debug_dir,
    }

    log_event(logger, f"[INFO] processing user requests: {user_requests_with_active_tasks}")
    model = methods_dict[args.method](args).to(args.device)
    init_model = model.state_dict()
    model.load_state_dict(init_model)
    current_stat = process_requests(
        args,
        model,
        train_datasets,
        test_datasets,
        user_requests_with_active_tasks,
        run_context,
    )

    if not metrics_state.get("unlearning_events"):
        msg = (
            "[INFO] No unlearning events in this run; compare against pall_original to "
            "confirm CL training does not regress."
        )
        metrics_state.setdefault("sanity_checks", []).append(msg)
        write_json(metrics_path, metrics_state)
        log_event(logger, msg)

    result = {
        'stats': current_stat,
        'user_requests_with_active_tasks': user_requests_with_active_tasks,
        'user_requests_without_forgotten': user_requests_without_forgotten,
    }

    torch.save(result, run_dir / "results.pth")

    if args.dump_overlap and hasattr(model, "compute_overlap_matrix"):
        overlap = model.compute_overlap_matrix(include_forgotten=True)
        task_ids = overlap.get("task_ids", [])
        matrix = overlap.get("matrix", [])
        if task_ids and matrix:
            write_overlap_csv(run_dir / "overlap.csv", task_ids, matrix)
            log_event(logger, f"[INFO] wrote overlap matrix to {run_dir / 'overlap.csv'}")

    final_acc = []
    if len(current_stat["accuracy"]) > 0:
        final_acc = to_list(current_stat["accuracy"][-1])
    final_active_tasks = user_requests_with_active_tasks[-1][2] if user_requests_with_active_tasks else []
    final_avg = avg_for_tasks(final_acc, final_active_tasks)

    summary = {
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "method": args.method,
        "method_variant": args.method_variant,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "n_tasks": args.n_tasks,
        "n_forget": args.n_forget,
        "final_avg_accuracy": final_avg,
    }
    metrics_state["summary"] = summary
    write_json(metrics_path, metrics_state)
    write_summary(run_dir / "summary.txt", summary, metrics_state.get("unlearning_events", []))


if __name__ == "__main__":
    main()
