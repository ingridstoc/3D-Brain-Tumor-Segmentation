from __future__ import annotations

import copy
import json
import os
import random
from datetime import datetime

import yaml

from train import main, CFG


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_optimizer_params(cfg_dict: dict, optimizer_name: str, lr: float, weight_decay: float):
    cfg_dict["optimizer"]["name"] = optimizer_name

    if optimizer_name == "adamw":
        cfg_dict["optimizer"]["adamw"]["lr"] = lr
        cfg_dict["optimizer"]["adamw"]["weight_decay"] = weight_decay

    elif optimizer_name == "adam":
        cfg_dict["optimizer"]["adam"]["lr"] = lr
        cfg_dict["optimizer"]["adam"]["weight_decay"] = weight_decay

    elif optimizer_name == "rmsprop":
        cfg_dict["optimizer"]["rmsprop"]["lr"] = lr
        cfg_dict["optimizer"]["rmsprop"]["weight_decay"] = weight_decay

    else:
        raise ValueError(f"Unsupported optimizer in search: {optimizer_name}")


def set_scheduler_params(cfg_dict: dict, scheduler_name: str, fixed_scheduler_params: dict):
    cfg_dict["scheduler"]["name"] = scheduler_name
    if scheduler_name not in fixed_scheduler_params:
        raise ValueError(f"Missing fixed params for scheduler: {scheduler_name}")
    cfg_dict["scheduler"][scheduler_name] = copy.deepcopy(fixed_scheduler_params[scheduler_name])


def set_loss_params(cfg_dict: dict, loss_name: str, fixed_loss_params: dict):
    if loss_name not in fixed_loss_params:
        raise ValueError(f"Missing fixed params for loss: {loss_name}")

    cfg_dict["loss"] = {"name": loss_name}
    cfg_dict["loss"].update(copy.deepcopy(fixed_loss_params[loss_name]))


def set_augmentation(cfg_dict: dict, augmentation_name: str):
    cfg_dict["augmentations"]["name"] = augmentation_name


def sample_trial_config(base_cfg: dict, trial_idx: int) -> dict:
    cfg_dict = copy.deepcopy(base_cfg)

    rs = cfg_dict["random_search"]
    space = rs["search_space"]

    optimizer_name = random.choice(space["optimizer_names"])
    lr = random.choice(space["lr"])
    weight_decay = random.choice(space["weight_decay"])
    loss_name = random.choice(space["loss_names"])
    scheduler_name = random.choice(space["scheduler_names"])
    augmentation_name = random.choice(space["augmentation_names"])

    set_optimizer_params(cfg_dict, optimizer_name, lr, weight_decay)
    set_scheduler_params(cfg_dict, scheduler_name, rs["fixed_scheduler_params"])
    set_loss_params(cfg_dict, loss_name, rs["fixed_loss_params"])
    set_augmentation(cfg_dict, augmentation_name)

    run_name = (
        f"{cfg_dict['modality']}_trial_{trial_idx:03d}_"
        f"opt-{optimizer_name}_lr-{lr}_wd-{weight_decay}_"
        f"loss-{loss_name}_sch-{scheduler_name}_aug-{augmentation_name}"
    )
    cfg_dict["run_name"] = run_name

    return cfg_dict


def append_jsonl(path: str, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_random_search(config_path: str = "config.yaml"):
    base_cfg = load_yaml(config_path)
    rs_cfg = base_cfg.get("random_search", {})

    if not rs_cfg.get("enabled", False):
        raise ValueError("random_search.enabled is false in config.yaml")

    n_trials = rs_cfg.get("n_trials", 10)
    save_dir = rs_cfg.get("save_dir", "random_search_results")
    modality = base_cfg["modality"]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, modality), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_jsonl = os.path.join(save_dir, modality, f"results_{timestamp}.jsonl")
    best_json = os.path.join(save_dir, modality, f"best_{timestamp}.json")

    best_result = None

    print(f"Starting random search for modality={modality}")
    print(f"Trials: {n_trials}")
    print(f"Results file: {results_jsonl}")

    for trial_idx in range(1, n_trials + 1):
        print("\n" + "=" * 80)
        print(f"TRIAL {trial_idx}/{n_trials}")

        trial_cfg_dict = sample_trial_config(base_cfg, trial_idx)
        trial_cfg_path = os.path.join(save_dir, modality, f"{trial_cfg_dict['run_name']}.yaml")
        save_yaml(trial_cfg_path, trial_cfg_dict)
        cfg = CFG(trial_cfg_dict)

        result = main(cfg)

        row = {
            "trial_idx": trial_idx,
            "run_name": trial_cfg_dict["run_name"],
            "modality": cfg.modality,
            "optimizer_name": cfg.optimizer_name,
            "optimizer_params": cfg.optimizer_params,
            "scheduler_name": cfg.scheduler_name,
            "scheduler_params": cfg.scheduler_params,
            "loss_cfg": cfg.loss_cfg,
            "augmentation_name": cfg.augmentation_name,
            "best_epoch": result["best_epoch"],
            "best_val_dice": result["best_val_dice"],
            "best_val_loss": result["best_val_loss"],
            "best_val_pc": result["best_val_pc"],
            "final_val_dice": result["final_val_dice"],
            "final_val_loss": result["final_val_loss"],
        }

        append_jsonl(results_jsonl, row)

        if best_result is None or row["best_val_dice"] > best_result["best_val_dice"]:
            best_result = row
            with open(best_json, "w") as f:
                json.dump(best_result, f, indent=2)

        print("\nCurrent best:")
        print(json.dumps(best_result, indent=2))

    print("\n" + "=" * 80)
    print("RANDOM SEARCH DONE")
    print("Best trial:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    run_random_search("config.yaml")