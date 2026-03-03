
import random
import numpy as np
import torch
from monai.losses import DiceCELoss

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Loss
# -------------------------
def make_loss():
    # Expects logits [B,C,*] and label [B,*] with integer class ids 0..C-1
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,      # applies softmax to logits internally for Dice part
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


from dataclasses import dataclass
@dataclass
class CFG:
    root: str  # directory with patient folders
    num_classes: int = 4  # internal: {0,1,2,3} where 3==original 4
    batch_size: int = 1  # 128^3 is heavy; start with 1
    num_workers: int = 1
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42
    include_bg_in_metric: bool = False
    ensemble_temp: float = 1.0  # softmax temperature for class-wise weights
    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = make_loss()



