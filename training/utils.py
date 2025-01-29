import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    train_transforms: transforms.Compose,
    val_transforms: transforms.Compose,
    batch_size: int,
):
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, class_names


def get_transforms(model_path: str):
    data_config = timm.data.resolve_model_data_config(model_path)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    val_transforms = timm.data.create_transform(**data_config, is_training=False)

    return train_transforms, val_transforms
