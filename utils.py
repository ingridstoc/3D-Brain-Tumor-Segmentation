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

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceCELoss, DiceFocalLoss, GeneralizedDiceLoss


class GeneralizedDiceCELoss(nn.Module):
    def __init__(
        self,
        to_onehot_y=True,
        softmax=True,
        lambda_gdice=1.0,
        lambda_ce=1.0,
    ):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.lambda_gdice = lambda_gdice
        self.lambda_ce = lambda_ce

        self.gdice = GeneralizedDiceLoss(
            to_onehot_y=to_onehot_y,
            softmax=softmax,
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        gdice_loss = self.gdice(logits, target)

        if target.ndim == 5 and target.shape[1] == 1:
            ce_target = target[:, 0].long()
        else:
            ce_target = target.long()

        ce_loss = self.ce(logits, ce_target)

        return self.lambda_gdice * gdice_loss + self.lambda_ce * ce_loss

import yaml
import torch


class CFG:
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, "r") as f:
                data = yaml.safe_load(f)
        elif isinstance(source, dict):
            data = source
        else:
            raise ValueError("CFG source must be a yaml path or a dict")


        self.raw = data
        self.model_name = data.get("model", {}).get("name", "unet").lower()
        self.root = data["root"]
        self.modality = data.get("modality", "t1").lower()
        self.run_name = data.get("run_name", self.modality)
        self.num_classes = data.get("num_classes", 4)
        self.batch_size = data.get("batch_size", 1)
        self.num_workers = data.get("num_workers", 1)
        self.epochs = data.get("epochs", 1)
        self.seed = data.get("seed", 42)
        self.include_bg_in_metric = data.get("include_bg_in_metric", False)

        device_value = data.get("device", "auto")
        if device_value == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_value)

        optimizer_cfg = data.get("optimizer", {})
        self.optimizer_name = optimizer_cfg.get("name", "adamw").lower()
        self.optimizer_params = optimizer_cfg.get(self.optimizer_name, {})

        scheduler_cfg = data.get("scheduler", {})
        self.scheduler_name = scheduler_cfg.get("name", "none").lower()
        self.scheduler_params = scheduler_cfg.get(self.scheduler_name, {})

        
        loss_cfg = data.get("loss", {})
        self.loss_name = loss_cfg.get("name", "dice_ce").lower()
        self.loss_params = loss_cfg.get(self.loss_name, {})
        self.loss_params["name"] = self.loss_name
        self.loss_fn = make_loss(self.loss_params)

        aug_cfg = data.get("augmentations", {})
        self.augmentation_name = aug_cfg.get("name", "none").lower()
    def print_parameters(self) -> dict:
        optimizer_dict = dict(self.optimizer_params)
        optimizer_dict["name"] = self.optimizer_name

        scheduler_dict = dict(self.scheduler_params)
        scheduler_dict["name"] = self.scheduler_name

        loss_dict = dict(self.loss_params)
        
        return {
            "root": self.root,
            "modality": self.modality,
            "run_name": self.run_name,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "seed": self.seed,
            "include_bg_in_met": self.include_bg_in_metric,
            "device": str(self.device),
            "optimizer": optimizer_dict,
            "scheduler": scheduler_dict,
            "loss": loss_dict,
            "augmentations": {
                "name": self.augmentation_name
            },
            "model": {
                "name": self.model_name
            }
        }


def make_loss(loss_cfg: dict):
    loss_name = loss_cfg["name"].lower()

    if loss_name == "dice_ce":
        return DiceCELoss(
            to_onehot_y=loss_cfg.get("to_onehot_y", True),
            softmax=loss_cfg.get("softmax", True),
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
            lambda_ce=loss_cfg.get("lambda_ce", 1.0),
        )

    if loss_name == "dice_focal":
        return DiceFocalLoss(
            to_onehot_y=loss_cfg.get("to_onehot_y", True),
            softmax=loss_cfg.get("softmax", True),
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            gamma=loss_cfg.get("gamma", 2.0),
        )

    if loss_name == "generalized_dice_ce":
        return GeneralizedDiceCELoss(
            to_onehot_y=loss_cfg.get("to_onehot_y", True),
            softmax=loss_cfg.get("softmax", True),
            lambda_gdice=loss_cfg.get("lambda_gdice", 1.0),
            lambda_ce=loss_cfg.get("lambda_ce", 1.0),
        )

    raise ValueError(f"Unknown loss name: {loss_name}")


def build_optimizer(cfg, model):
    optimizer_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
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
        # "step": torch.optim.lr_scheduler.StepLR,
        # "multistep": torch.optim.lr_scheduler.MultiStepLR,
        # "exponential": torch.optim.lr_scheduler.ExponentialLR,
        # "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        # "cosine_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "none": None,
    }

    if cfg.scheduler_name not in scheduler_map:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler_name}")

    scheduler_class = scheduler_map[cfg.scheduler_name]

    if scheduler_class is None:
        return None

    return scheduler_class(optimizer, **cfg.scheduler_params)