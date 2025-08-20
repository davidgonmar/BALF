from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
import torch


def load_model(
    model_name: str, pretrained_path: str = None, extra_args={}
) -> torch.nn.Module:
    model_dict = {
        "resnet20": resnet20,
        "resnet32": resnet32,
        "resnet44": resnet44,
        "resnet56": resnet56,
        "resnet110": resnet110,
        "resnet1202": resnet1202,
    }
    if model_name not in model_dict:
        raise ValueError(
            f"Model {model_name} is not supported. Choose from {list(model_dict.keys())}."
        )

    model = model_dict[model_name](**extra_args)
    if pretrained_path:
        loaded = torch.load(pretrained_path, weights_only=False)
        if isinstance(loaded, torch.nn.Module):
            model.load_state_dict(loaded.state_dict())
        else:
            model.load_state_dict(loaded)
    return model
