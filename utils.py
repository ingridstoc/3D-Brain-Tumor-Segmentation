import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.losses import DiceCELoss


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loss(loss_cfg: dict):
    loss_name = loss_cfg["name"].lower()

    if loss_name == "dice_ce":
        return DiceCELoss(
            to_onehot_y=loss_cfg.get("to_onehot_y", True),
            softmax=loss_cfg.get("softmax", True),
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
            lambda_ce=loss_cfg.get("lambda_ce", 1.0),
        )

    raise ValueError(f"Unknown loss name: {loss_name}")

class CFG:
    def __init__(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # basic params
        self.root = data["root"]
        self.num_classes = data.get("num_classes", 4)
        self.batch_size = data.get("batch_size", 1)
        self.num_workers = data.get("num_workers", 1)
        self.epochs = data.get("epochs", 1)
        self.seed = data.get("seed", 42)
        self.include_bg_in_metric = data.get("include_bg_in_metric", False)

        # device
        device_value = data.get("device", "auto")
        if device_value == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_value)

        # optimizer config
        optimizer_cfg = data.get("optimizer", {})
        self.optimizer_name = optimizer_cfg.get("name", "adamw").lower()
        self.optimizer_params = optimizer_cfg.get(self.optimizer_name, {})

        # scheduler config
        scheduler_cfg = data.get("scheduler", {})
        self.scheduler_name = scheduler_cfg.get("name", "none").lower()
        self.scheduler_params = scheduler_cfg.get(self.scheduler_name, {})

        # loss config
        self.loss_cfg = data.get("loss", {"name": "dice_ce"})
        self.loss_fn = make_loss(self.loss_cfg)


def build_optimizer(cfg, model):
    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adamax": torch.optim.Adamax,
    }

    if cfg.optimizer_name not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")

    optimizer_class = optimizer_map[cfg.optimizer_name]
    return optimizer_class(model.parameters(), **cfg.optimizer_params)


def build_scheduler(cfg, optimizer):
    scheduler_map = {
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "step": torch.optim.lr_scheduler.StepLR,
        "multistep": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "none": None,
    }

    if cfg.scheduler_name not in scheduler_map:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler_name}")

    scheduler_class = scheduler_map[cfg.scheduler_name]

    if scheduler_class is None:
        return None

    return scheduler_class(optimizer, **cfg.scheduler_params)