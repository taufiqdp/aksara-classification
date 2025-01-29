import timm
from typing import Tuple
import torch.nn as nn
from torchvision import transforms

model_path = "hf_hub:taufiqdp/convnext_tiny-arutala"


def load_model(model_path: str) -> Tuple[nn.Module, transforms.Compose]:
    model = timm.create_model(model_path, pretrained=True)
    data_config = timm.data.resolve_model_data_config(model_path)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return model, transforms
