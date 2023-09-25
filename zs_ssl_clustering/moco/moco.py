import os

import torch
import torchvision.models as torchvision_models

from zs_ssl_clustering.moco import vits

model_name_to_weights = {
    "resnet50": "r-50-1000ep.pth.tar",
    "vit_small": "vit-s-300ep.pth.tar",
    "vit_base": "vit-b-300ep.pth.tar",
}


def load_pretrained_model(model_name, pretrained_dir):
    if model_name.startswith("vit"):
        model = vits.__dict__[model_name]()
        linear_keyword = "head"
    else:
        model = torchvision_models.__dict__[model_name]()
        linear_keyword = "fc"

    weight_path = os.path.join(pretrained_dir, model_name_to_weights[model_name])
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
