from dataclasses import dataclass


@dataclass
class Config:
    # Model settings
    model_path: str = "timm/convnext_tiny.in12k_ft_in1k"

    # Training settings
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 5e-5
    weight_decay: float = 0.001

    # Data paths
    train_dir: str = "data/train"
    val_dir: str = "data/test"

    # Device settings
    device: str = "cuda"
    use_amp: bool = True

    # Other settings
    model_save_dir: str = "models"
