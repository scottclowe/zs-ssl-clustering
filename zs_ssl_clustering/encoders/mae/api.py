import os
import urllib.request
from inspect import getsourcefile

import torch

from . import models_vit

model_name_to_url = {
    "pretrain_vit_base": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
    "pretrain_vit_large": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
    "pretrain_vit_huge": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
    "finetuned_vit_base": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
    "finetuned_vit_large": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth",
    "finetuned_vit_huge": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth",
}
model_name_to_weights = {k: v.split("/")[-1] for k, v in model_name_to_url.items()}

MAE_SOURCE_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
DEFAULT_CACHE = os.path.join(MAE_SOURCE_DIR, ".cache")


def load_pretrained_model(
    model_name,
    pretrained_dir=DEFAULT_CACHE,
    download=True,
    drop_path=0.0,
    global_pool=None,
):
    # Default with the SSL pretrained model
    if model_name.startswith("vit_"):
        model_name = "pretrain_" + model_name

    # Support global token pooling or cls (not token pooling) being specified in the model name
    if model_name.endswith("_global"):
        global_pool = True
        model_name = model_name[:-7]
    if model_name.endswith("_cls"):
        global_pool = False
        model_name = model_name[:-4]

    model_name = model_name.replace("_base_patch16", "_base")
    model_name = model_name.replace("_large_patch16", "_large")
    model_name = model_name.replace("_huge_patch14", "_huge")

    if model_name not in model_name_to_weights:
        raise ValueError(
            f"Unrecognized MAE model '{model_name}'. Available models are: "
            f"{list(model_name_to_weights.keys())}."
        )

    print(f"Loading MAE model '{model_name}', with global_pool={global_pool}")
    if "vit_base" in model_name:
        model = models_vit.vit_base_patch16(
            num_classes=0,
            drop_path_rate=drop_path,
            global_pool=global_pool,
        )
    elif "vit_large" in model_name:
        model = models_vit.vit_large_patch16(
            num_classes=0,
            drop_path_rate=drop_path,
            global_pool=global_pool,
        )
    elif "vit_huge" in model_name:
        model = models_vit.vit_huge_patch14(
            num_classes=0,
            drop_path_rate=drop_path,
            global_pool=global_pool,
        )
    else:
        raise ValueError(f"Unrecognized MAE model '{model_name}'.")

    weight_path = os.path.join(pretrained_dir, model_name_to_weights[model_name])

    if not os.path.isfile(weight_path) and download:
        print(
            f"Downloading weights from\n\t{model_name_to_url[model_name]}"
            f"\n\tto {weight_path}"
        )
        os.makedirs(pretrained_dir, exist_ok=True)
        urllib.request.urlretrieve(model_name_to_url[model_name], weight_path)

    assert os.path.isfile(weight_path)

    print(f"=> loading checkpoint '{weight_path}'")
    checkpoint = torch.load(weight_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint["model"]
    state_dict = {k: v for (k, v) in state_dict.items() if not k.startswith("head.")}

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    if not msg.missing_keys:
        print("Successfully loaded pre-trained model")

    acceptable_missing_keys = {"head.weight", "head.bias"}
    if global_pool:
        acceptable_missing_keys.update({"fc_norm.weight", "fc_norm.bias"})
    else:
        acceptable_missing_keys.update({"norm.weight", "norm.bias"})

    assert len(set(msg.missing_keys).difference(acceptable_missing_keys)) == 0

    print("=> loaded pre-trained model '{}'".format(weight_path))

    return model
