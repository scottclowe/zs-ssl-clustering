import os
import warnings
from inspect import getsourcefile

import timm
import torch
import torchvision
from timm.data import resolve_data_config
from torch import nn

from . import mae, moco

PACKAGE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
)
REPO_DIR = os.path.dirname(PACKAGE_DIR)


def get_timm_encoder(model_name, pretrained=False, in_chans=3):
    r"""
    Get the encoder model and its configuration from timm.

    Parameters
    ----------
    model_name : str
        Name of the model to load.
    pretrained : bool, default=False
        Whether to load the model with pretrained weights.
    in_chans : int, default=3
        Number of input channels.

    Returns
    -------
    encoder : torch.nn.Module
        The encoder model (with pretrained weights loaded if requested).
    encoder_config : dict
        The data configuration of the encoder model.
    """
    if len(timm.list_models(model_name)) == 0:
        warnings.warn(
            f"Unrecognized model '{model_name}'. Trying to fetch it from the"
            " hugging-face hub.",
            UserWarning,
            stacklevel=2,
        )
        model_name = "hf-hub:timm/" + model_name

    # We request the model without the classification head (num_classes=0)
    # to get it is an encoder-only model
    encoder = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
    )
    encoder_config = resolve_data_config({}, model=encoder)

    # Send a dummy input through the encoder to find out the shape of its output
    encoder.eval()
    dummy_output = encoder(torch.zeros((1, *encoder_config["input_size"])))
    encoder_config["n_feature"] = dummy_output.shape[1]
    encoder.train()

    encoder_config["in_channels"] = encoder_config["input_size"][0]

    return encoder, encoder_config


class TorchVisionEncoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True):
        super().__init__()
        weights = None
        if model_name == "resnet18":
            if pretrained:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            self.model = torchvision.models.resnet18(weights=weights)
        elif model_name == "resnet50":
            if pretrained:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            self.model = torchvision.models.resnet50(weights=weights)
        elif model_name in ["vit_b_16", "vitb16"]:
            if pretrained:
                weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            self.model = torchvision.models.vit_b_16(weights=weights)
        else:
            raise ValueError(f"Unrecognized model: '{model_name}'.")
        if "resnet" in model_name:
            self.model.fc = nn.Identity()
        elif "vit" in model_name:
            self.model.heads = nn.Identity()

    def forward(self, x):
        return self.model(x)


class TimmEncoder(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        model_name = model_name.replace("timm_", "")
        self.model, self.data_config = get_timm_encoder(model_name, pretrained=True)

    def forward(self, x):
        return self.model(x)


class DINOv2(nn.Module):
    def __init__(self, model_name="vits14"):
        super().__init__()
        if "dinov2_" not in model_name:
            model_name = "dinov2_" + model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)

    def forward(self, x):
        return self.model(x)


class DINOv1(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        if "dino_" not in model_name:
            model_name = "dino_" + model_name
        self.model = torch.hub.load("facebookresearch/dino:main", model_name)

    def forward(self, x):
        return self.model(x)


class SWAV(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        model_name = model_name.replace("swav_", "")
        self.model = torch.hub.load("facebookresearch/swav:main", model_name)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class BarlowTwins(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        model_name = model_name.replace("barlowtwins_", "")
        self.model = torch.hub.load("facebookresearch/barlowtwins:main", model_name)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class VICReg(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        model_name = model_name.replace("vicreg_", "")
        self.model = torch.hub.load("facebookresearch/vicreg:main", model_name)

    def forward(self, x):
        return self.model(x)


class VICRegL(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        model_name = model_name.replace("vicregl_", "")
        self.model = torch.hub.load("facebookresearch/vicregl:main", model_name)

    def forward(self, x):
        return self.model(x)[-1]


class CLIP(nn.Module):
    def __init__(self, model_name="RN50"):
        super().__init__()

        import clip

        model_name = model_name.replace("clip_", "")
        # Flexible handling of the format for model names
        _model_name = model_name.lower().replace("-", "").replace("/", "")
        if _model_name == "vitb16":
            model_name = "ViT-B/16"
        elif _model_name == "vitb32":
            model_name = "ViT-B/32"
        elif _model_name == "vitl14":
            model_name = "ViT-L/14"
        elif _model_name == "vitl14@336px":
            model_name = "ViT-L/14@336px"
        elif model_name:
            model_name = model_name.replace("resnet", "RN")
        self.model, self.transform = clip.load(model_name)

    def forward(self, x):
        return self.model.encode_image(x)


class MoCoV3(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()

        model_name = model_name.replace("mocov3_", "")

        self.model = moco.load_pretrained_model(model_name)
        if "vit" in model_name:
            self.model.head = nn.Identity()
        else:
            self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class MAE(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()

        if model_name.startswith("mae_"):
            model_name = model_name[4:]
        self.model = mae.load_pretrained_model(model_name)
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)


def get_encoder(model_name):
    if model_name.startswith("ft_"):
        return get_finetuned_encoder(model_name)

    elif model_name.startswith("timm"):
        return TimmEncoder(model_name)

    elif model_name.startswith("dinov2"):
        return DINOv2(model_name)

    elif model_name.startswith("dino"):
        return DINOv1(model_name)

    elif model_name.startswith("barlowtwins"):
        return BarlowTwins(model_name)

    elif model_name.startswith("swav"):
        return SWAV(model_name)

    elif model_name.startswith("vicregl"):
        return VICRegL(model_name)

    elif model_name.startswith("vicreg"):
        return VICReg(model_name)

    elif model_name.startswith("clip"):
        return CLIP(model_name)

    elif model_name.startswith("mocov3"):
        return MoCoV3(model_name)

    elif model_name.startswith("mae"):
        return MAE(model_name)

    elif model_name.startswith("random"):
        return TorchVisionEncoder(model_name[6:].strip("_"), pretrained=False)

    else:
        return TorchVisionEncoder(model_name)


def get_finetuned_encoder(model_name):
    ft_models_dir = os.path.join(REPO_DIR, "ft-models")

    if model_name in [
        "ft_mocov3_resnet50",
        "ft_mocov3_vit_base",
        "ft_dino_resnet50",
        "ft_dino_vitb16",
    ]:
        encoder = get_encoder(model_name[3:])
        fname = os.path.join(ft_models_dir, model_name + ".pth")
        print("Loading fine-tuned model from:", fname)
        state_dict = torch.load(fname, map_location="cpu")
        state_dict = state_dict["model"]
        state_dict = {
            key: value
            for (key, value) in state_dict.items()
            if not key.startswith("fc.") and not key.startswith("head.")
        }
        encoder.model.load_state_dict(state_dict, strict=True)
        return encoder

    if model_name == "ft_vicreg_resnet50":
        encoder = get_encoder(model_name[3:])
        fname = os.path.join(ft_models_dir, model_name + ".pth")
        print("Loading fine-tuned model from:", fname)
        state_dict = torch.load(fname, map_location="cpu")
        state_dict = state_dict["model"]
        state_dict = {
            key[2:] if key.startswith("0.") else key: value
            for (key, value) in state_dict.items()
            if not key.startswith("1.")
            and not key.startswith("fc.")
            and not key.startswith("head.")
        }
        encoder.model.load_state_dict(state_dict, strict=True)
        return encoder

    raise NotImplementedError(f"Finetuned encoder '{model_name}' not implemented.")
