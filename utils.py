import random
import numpy as np
import torch
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
        self.ensemble_temp = data.get("ensemble_temp", 1.0)

        # device
        device_value = data.get("device", "auto")
        if device_value == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_value)

        # optimizer config
        optimizer_cfg = data.get("optimizer", {})
        self.optimizer_name = optimizer_cfg.get("name", "adamw").lower()

        # prefer optimizer section values if present
        self.lr = optimizer_cfg.get("lr", data.get("lr", 1e-4))
        self.weight_decay = optimizer_cfg.get("weight_decay", data.get("weight_decay", 1e-5))

        # loss config
        self.loss_cfg = data.get("loss", {"name": "dice_ce"})
        self.loss_fn = make_loss(self.loss_cfg)