# Ensemble prediction (class-wise weights)
# -------------------------
# @torch.no_grad()
# def ensemble_predict_classwise(models_by_name, W, names, x_by_name, device, num_classes=4):
#     """
#     models_by_name: dict modality->model
#     W: Tensor[M,C] class-wise weights, sum over modalities = 1 for each class
#     names: list modality names matching W rows
#     x_by_name: dict modality-> input tensor [B,1,H,W,D]
#     returns pred [B,H,W,D] in internal labels 0..3 (map 3->4 if needed)
#     """
#     p_final = None
#     for mi, name in enumerate(names):
#         model = models_by_name[name]
#         x = x_by_name[name].to(device)
#         logits = model(x)
#         probs = torch.softmax(logits, dim=1)  # [B,C,H,W,D]
#         w_mc = W[mi].to(device).view(1, num_classes, 1, 1, 1)
#         contrib = probs * w_mc
#         p_final = contrib if p_final is None else (p_final + contrib)

#     pred = torch.argmax(p_final, dim=1)
#     return pred


def compute_classwise_weights(dice_by_model: Dict[str, torch.Tensor], temp: float = 1.0, eps: float = 1e-6):
    """
    dice_by_model: dict modality -> Tensor[C]
    returns:
      names: list[str]
      W: Tensor[M,C] where sum over M is 1 for each class (softmax over modalities)
    """
    names = list(dice_by_model.keys())
    d = torch.stack([dice_by_model[n] for n in names], dim=0)  # [M,C]
    d = torch.nan_to_num(d, nan=0.0)
    scores = torch.log(d + eps) / temp
    W = torch.softmax(scores, dim=0)
    return names, W

@torch.no_grad()
def dice_per_class_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    include_bg: bool = False,
):
    model.eval()
    eps = 1e-6
    inter = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x = x.to(device)  # [B,1,H,W,D]
        y = y.to(device)  # [B,H,W,D]
        logits = model(x)
        pred = torch.argmax(logits, dim=1)  # [B,H,W,D]

        for c in range(num_classes):
            if (not include_bg) and c == 0:
                continue
            p = pred == c
            g = y == c
            inter[c] += (p & g).sum()
            denom[c] += p.sum() + g.sum()

    dice = (2.0 * inter + eps) / (denom + eps)  # [C]
    if not include_bg:
        dice[0] = torch.nan
    return dice.detach().cpu()