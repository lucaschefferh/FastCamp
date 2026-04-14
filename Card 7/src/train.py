"""
train.py
────────────────────────────────────────────────────────────────
Loop de treinamento para segmentação de tumor cerebral (LGG MRI).

Uso básico:
    python src/train.py

Uso com argumentos:
    python src/train.py --model unetpp --backbone efficientnet-b4 --epochs 80
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders


# ─────────────────────────────────────────────────────────────────
# Configurações padrão
# ─────────────────────────────────────────────────────────────────

DEFAULTS = {
    "dataset_dir":  "dataset",
    "checkpoints":  "checkpoints",
    "model":        "unet",           # unet | unetpp | deeplabv3p
    "backbone":     "resnet34",
    "epochs":       100,
    "batch_size":   16,
    "lr":           1e-4,
    "patience":     15,               # early stopping
    "num_workers":  4,
}


# ─────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────

def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Dice calculado sobre o batch inteiro, após binarização."""
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection / (preds.sum() + targets.sum() + 1e-7)).item()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection over Union calculado sobre o batch inteiro."""
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / (union + 1e-7)).item()


# ─────────────────────────────────────────────────────────────────
# Construção do modelo
# ─────────────────────────────────────────────────────────────────

MODEL_MAP = {
    "unet":       smp.Unet,
    "unetpp":     smp.UnetPlusPlus,
    "deeplabv3p": smp.DeepLabV3Plus,
}

def build_model(model_name: str, backbone: str) -> nn.Module:
    cls = MODEL_MAP.get(model_name)
    if cls is None:
        raise ValueError(f"Modelo '{model_name}' desconhecido. Opções: {list(MODEL_MAP)}")
    return cls(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,   # logits brutos — perda e métricas aplicam sigmoid internamente
    )


# ─────────────────────────────────────────────────────────────────
# Um epoch de treino / validação
# ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)

            # DiceLoss espera máscara (N, H, W) sem canal
            loss = criterion(preds, masks.squeeze(1))

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(preds, masks)
            total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    # Dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        cfg["dataset_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"Batches — train: {len(train_loader)} | val: {len(val_loader)}")

    # Modelo
    model = build_model(cfg["model"], cfg["backbone"]).to(device)
    print(f"Modelo: {cfg['model']} + {cfg['backbone']}\n")

    # Perda: Dice + BCE (trata desbalanceamento de classes)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss  = nn.BCEWithLogitsLoss()
    criterion = lambda preds, masks: 0.5 * dice_loss(preds, masks) + 0.5 * bce_loss(preds, masks)

    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Checkpoint
    ckpt_dir = Path(cfg["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['model']}_{cfg['backbone']}_best.pth"

    # Early stopping
    best_dice     = 0.0
    epochs_no_imp = 0
    history       = []

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Dice':>10} | {'Val Loss':>9} | {'Val Dice':>9} | {'Val IoU':>8} | {'LR':>8}")
    print("-" * 80)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_dice, _         = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_dice,   val_iou   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_dice": train_dice,
            "val_loss":   val_loss,   "val_dice":   val_dice, "val_iou": val_iou,
            "lr": current_lr,
        })

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_dice:>10.4f} | {val_loss:>9.4f} | {val_dice:>9.4f} | {val_iou:>8.4f} | {current_lr:>8.2e}  ({elapsed:.0f}s)")

        # Salva melhor checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_imp = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice":    val_dice,
                "val_iou":     val_iou,
                "cfg":         cfg,
            }, ckpt_path)
            print(f"         ✓ checkpoint salvo (val_dice={val_dice:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= cfg["patience"]:
                print(f"\nEarly stopping na época {epoch} (sem melhora por {cfg['patience']} épocas)")
                break

    print(f"\nTreino concluído. Melhor val_dice: {best_dice:.4f}")
    print(f"Checkpoint salvo em: {ckpt_path}")
    return history, ckpt_path


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Treino de segmentação LGG MRI")
    p.add_argument("--dataset_dir",  default=DEFAULTS["dataset_dir"])
    p.add_argument("--checkpoints",  default=DEFAULTS["checkpoints"])
    p.add_argument("--model",        default=DEFAULTS["model"],      choices=list(MODEL_MAP))
    p.add_argument("--backbone",     default=DEFAULTS["backbone"])
    p.add_argument("--epochs",       default=DEFAULTS["epochs"],     type=int)
    p.add_argument("--batch_size",   default=DEFAULTS["batch_size"], type=int)
    p.add_argument("--lr",           default=DEFAULTS["lr"],         type=float)
    p.add_argument("--patience",     default=DEFAULTS["patience"],   type=int)
    p.add_argument("--num_workers",  default=DEFAULTS["num_workers"],type=int)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
