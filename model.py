import timm


def load_model(model_path: str, num_classes: int, pretrained: bool = True):
    model = timm.create_model(
        model_path, pretrained=pretrained, num_classes=num_classes
    )
    return model
