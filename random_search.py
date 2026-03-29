# from __future__ import annotations

# import copy
# import json
# import os
# import random
# from datetime import datetime

# import yaml

# from train import main, CFG


# def load_yaml(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_yaml(path: str, data: dict):
#     with open(path, "w") as f:
#         yaml.safe_dump(data, f, sort_keys=False)


# def set_optimizer_params(cfg_dict: dict, optimizer_name: str, lr: float):
#     cfg_dict["optimizer"]["name"] = optimizer_name

#     if optimizer_name == "adamw":
#         cfg_dict["optimizer"]["adamw"]["lr"] = lr

#     elif optimizer_name == "adam":
#         cfg_dict["optimizer"]["adam"]["lr"] = lr

#     elif optimizer_name == "rmsprop":
#         cfg_dict["optimizer"]["rmsprop"]["lr"] = lr

#     else:
#         raise ValueError(f"Unsupported optimizer in search: {optimizer_name}")


# def sample_trial_config(base_cfg: dict, trial_idx: int) -> dict:
#     cfg_dict = copy.deepcopy(base_cfg)
#     optimizer_name = random.choice(["adamw", "adam", "rmsprop"])
#     lr = random.choice([0.0001, 0.0002, 0.0003, 0.0005, 0.001])
#     loss_name = random.choice(["dice_ce", "dice_focal", "generalized_dice_ce"])
#     scheduler_name = random.choice(["reduce_on_plateau", "cosine", "cosine_warm_restarts"])
    
#     cfg_dict["optimizer"]["name"] = optimizer_name
#     cfg_dict["optimizer"][optimizer_name]["lr"] = lr
#     cfg_dict["scheduler"]["name"] = scheduler_name
#     cfg_dict["loss"]["name"] = loss_name

#     run_name = (
#         f"{cfg_dict['modality']}_trial_{trial_idx:03d}_"
#         f"opt-{optimizer_name}_lr-{lr}_"
#         f"loss-{loss_name}_sch-{scheduler_name}"
#     )

#     return cfg_dict, run_name


# def append_jsonl(path: str, row: dict):
#     with open(path, "a") as f:
#         f.write(json.dumps(row) + "\n")


# def run_random_search(n_trials : int = 16, config_path: str = "config.yaml"):
#     base_cfg = load_yaml(config_path)
    
#     save_dir = "random_search_results"
#     modality = base_cfg["modality"]

#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(os.path.join(save_dir, modality), exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_jsonl = os.path.join(save_dir, modality, f"results_{timestamp}.jsonl")
#     best_json = os.path.join(save_dir, modality, f"best_{timestamp}.json")

#     best_result = None

#     print(f"Starting random search for modality={modality}")
#     print(f"Trials: {n_trials}")
#     print(f"Results file: {results_jsonl}")

#     for trial_idx in range(1, n_trials + 1):
#         print("\n" + "=" * 80)
#         print(f"TRIAL {trial_idx}/{n_trials}")

#         trial_cfg_dict, config_name = sample_trial_config(base_cfg, trial_idx)
#         trial_cfg_path = os.path.join(save_dir, modality, f"{config_name}.yaml")
#         cfg = CFG(trial_cfg_dict)
#         save_yaml(trial_cfg_path, cfg.print_parameters())
#         print(config_name)
#         result = main(cfg)
#         result["config_name"] = config_name
#         append_jsonl(results_jsonl, result)

#         if best_result is None or result["best_val_dice"] > best_result["best_val_dice"]:
#             best_result = result
#             with open(best_json, "w") as f:
#                 json.dump(best_result, f, indent=2)

#         print("\nCurrent best:")
#         print(json.dumps(best_result, indent=2))

#     print("\n" + "=" * 80)
#     print("RANDOM SEARCH DONE")
#     print("Best trial:")
#     print(json.dumps(best_result, indent=2))


# if __name__ == "__main__":
#     run_random_search(n_trials=16, config_path="config.yaml")



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


def sample_trial_config(base_cfg: dict, trial_idx: int, rng: random.Random) -> tuple[dict, str]:
    cfg_dict = copy.deepcopy(base_cfg)

    optimizer_name = rng.choice(["adamw", "adam", "rmsprop"])
    lr = rng.choice([0.0001, 0.0002, 0.0003, 0.0005, 0.001])
    loss_name = rng.choice(["dice_ce", "dice_focal", "generalized_dice_ce"])
    scheduler_name = rng.choice(["reduce_on_plateau", "cosine", "cosine_warm_restarts"])
    augmentation_name = rng.choice(["none", "light", "medium", "strong"])

    cfg_dict["optimizer"]["name"] = optimizer_name
    cfg_dict["optimizer"][optimizer_name]["lr"] = lr

    cfg_dict["scheduler"]["name"] = scheduler_name

    cfg_dict["loss"]["name"] = loss_name

    if "augmentations" not in cfg_dict:
        cfg_dict["augmentations"] = {}
    cfg_dict["augmentations"]["name"] = augmentation_name

    run_name = (
        f"{cfg_dict['modality']}_trial_{trial_idx:03d}_"
        f"opt-{optimizer_name}_lr-{lr}_"
        f"loss-{loss_name}_sch-{scheduler_name}_aug-{augmentation_name}"
    )

    cfg_dict["run_name"] = run_name

    return cfg_dict, run_name


def append_jsonl(path: str, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def make_signature(cfg_dict: dict):
    opt_name = cfg_dict["optimizer"]["name"]
    sch_name = cfg_dict["scheduler"]["name"]
    loss_name = cfg_dict["loss"]["name"]
    aug_name = cfg_dict.get("augmentations", {}).get("name", "none")

    return (
        opt_name,
        cfg_dict["optimizer"][opt_name]["lr"],
        sch_name,
        loss_name,
        aug_name,
    )


def run_random_search(n_trials: int = 16, config_path: str = "config.yaml"):
    base_cfg = load_yaml(config_path)

    save_dir = "random_search_results"
    modality = base_cfg["modality"]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, modality), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_jsonl = os.path.join(save_dir, modality, f"results_{timestamp}.jsonl")
    best_json = os.path.join(save_dir, modality, f"best_{timestamp}.json")

    best_result = None

    rng = random.Random(12345)   # local RNG, not affected by seed_everything()
    seen_configs = set()

    print(f"Starting random search for modality={modality}")
    print(f"Trials: {n_trials}")
    print(f"Results file: {results_jsonl}")

    for trial_idx in range(1, n_trials + 1):
        print("\n" + "=" * 80)
        print(f"TRIAL {trial_idx}/{n_trials}")

        for _ in range(100):
            trial_cfg_dict, config_name = sample_trial_config(base_cfg, trial_idx, rng)
            sig = make_signature(trial_cfg_dict)
            if sig not in seen_configs:
                seen_configs.add(sig)
                break
        else:
            print("Could not find a new unique configuration.")
            break

        trial_cfg_path = os.path.join(save_dir, modality, f"{config_name}.yaml")

        cfg = CFG(trial_cfg_dict)
        save_yaml(trial_cfg_path, cfg.print_parameters())

        print(config_name)

        result = main(cfg)
        result["config_name"] = config_name
        append_jsonl(results_jsonl, result)

        if best_result is None or result["best_val_dice"] > best_result["best_val_dice"]:
            best_result = result
            with open(best_json, "w") as f:
                json.dump(best_result, f, indent=2)

        print("\nCurrent best:")
        print(json.dumps(best_result, indent=2))

    print("\n" + "=" * 80)
    print("RANDOM SEARCH DONE")
    print("Best trial:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    run_random_search(n_trials=16, config_path="config.yaml")