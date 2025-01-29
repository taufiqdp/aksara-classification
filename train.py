import torch
import os
from torch import nn
from huggingface_hub import login
from dotenv import load_dotenv
from .model import load_model
from .config import Config
from .utils import create_dataloaders, get_transforms
from .engine import train


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
config = Config()

train_dataloader, val_dataloader, class_names = create_dataloaders(
    config.train_dir,
    config.val_dir,
    *get_transforms(config.model_path),
    config.batch_size
)

model = load_model(config.model_path, num_classes=len(class_names))

param_groups = []
decay = []
no_decay = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if len(param.shape) == 1 or name.endswith(".bias"):
        no_decay.append(param)
    else:
        decay.append(param)

param_groups.append({"params": no_decay, "weight_decay": 0.0})
param_groups.append({"params": decay, "weight_decay": 0.001})

optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
loss_fn = nn.CrossEntropyLoss()

torch.compile(model)

result = train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    epochs=config.epochs,
    use_amp=config.use_amp,
    device=config.device,
)
