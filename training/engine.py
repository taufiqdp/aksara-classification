import os
from typing import Dict, List, Tuple

import torch
from config import Config
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

config = Config()


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    """Performs a single training step."""
    model.to(device)
    model.train()
    train_loss = 0

    data_loader_iter = tqdm(
        enumerate(data_loader), desc="Training", unit="step", total=len(data_loader)
    )

    for batch, (X, y) in data_loader_iter:
        X, y = X.to(device), y.to(device)

        with torch.autocast(enabled=use_amp, device_type=device, dtype=torch.float16):
            logits = model(X)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        data_loader_iter.set_postfix({"Train Loss": f"{train_loss / (batch + 1):.4f}"})

    return train_loss / len(data_loader)


def val_step(
    model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: str
) -> Tuple[float, float]:
    """Performs a single validation step."""
    model.to(device)
    model.eval()
    val_loss, val_acc = 0, 0

    data_loader_iter = tqdm(
        enumerate(data_loader), desc="Validation", unit="step", total=len(data_loader)
    )

    with torch.inference_mode():
        for batch, (X_val, y_val) in data_loader_iter:
            X_val, y_val = X_val.to(device), y_val.to(device)

            logits = model(X_val)
            loss = loss_fn(logits, y_val)
            val_loss += float(loss)

            val_pred_class = logits.argmax(dim=1)
            acc = (val_pred_class == y_val).sum().item() / len(val_pred_class)
            val_acc += acc

            data_loader_iter.set_postfix(
                {
                    "Val Loss": f"{val_loss / (batch + 1):.4f}",
                    "Val Acc": f"{val_acc / (batch + 1):.4f}",
                }
            )

    return val_loss / len(data_loader), val_acc / len(data_loader)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: str,
    use_amp: bool,
) -> Dict[str, List]:
    """Trains the model over the given epochs."""

    os.makedirs(config.model_save_dir, exist_ok=True)
    result = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    scaler = GradScaler(device)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} | lr: {scheduler.get_last_lr()}")
        result["lr"].append(scheduler.get_last_lr())

        train_loss = train_step(
            model, train_dataloader, loss_fn, optimizer, device, scaler, use_amp
        )
        result["train_loss"].append(train_loss)

        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, device)
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)

        scheduler.step()
        torch.save(model.state_dict(), f"{config.model_save_dir}/model-{epoch}.pth")

    return result
