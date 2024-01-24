import os
import urllib.request
from inspect import getsourcefile

import torch
import torchvision.models

from . import vits

model_name_to_weights = {
    "resnet50": "r-50-1000ep.pth.tar",
    "vit_small": "vit-s-300ep.pth.tar",
    "vit_base": "vit-b-300ep.pth.tar",
}
model_name_to_url = {
    "resnet50": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
    "vit_small": "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
    "vit_base": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
}

MOCO_SOURCE_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
DEFAULT_CACHE = os.path.join(MOCO_SOURCE_DIR, ".cache")


def load_pretrained_model(model_name, pretrained_dir=DEFAULT_CACHE, download=True):
    if model_name not in model_name_to_weights:
        raise ValueError(
            f"Unrecognized MoCo-v3 model '{model_name}'. Available models are: "
            f"{list(model_name_to_weights.keys())}."
        )

    if model_name.startswith("vit"):
        model = vits.__dict__[model_name]()
        linear_keyword = "head"
    else:
        model = torchvision.models.__dict__[model_name]()
        linear_keyword = "fc"

    weight_path = os.path.join(pretrained_dir, model_name_to_weights[model_name])

    if not os.path.isfile(weight_path) and download:
        print(
            f"Downloading weights from\n\t{model_name_to_url[model_name]}"
            f"\n\tto {weight_path}"
        )
        os.makedirs(pretrained_dir, exist_ok=True)
        urllib.request.urlretrieve(model_name_to_url[model_name], weight_path)

    assert os.path.isfile(weight_path)

    print("=> loading checkpoint '{}'".format(weight_path))
    checkpoint = torch.load(weight_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not k.startswith(
            "module.base_encoder.%s" % linear_keyword
        ):
            # remove prefix
            state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {
        "%s.weight" % linear_keyword,
        "%s.bias" % linear_keyword,
    }

    print("=> loaded pre-trained model '{}'".format(weight_path))

    return model
