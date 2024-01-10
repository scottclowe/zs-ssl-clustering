import torch
import torch.nn as nn
import vicreg.resnet as resnet
from timm.models.layers import trunc_normal_


def get_vicreg_model(args):
    model, emb_dim = resnet.__dict__[args.model](zero_init_residual=True)
    state_dict = torch.load(args.finetune, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(emb_dim, 1000)

    # manually initialize fc layer
    trunc_normal_(model.fc.weight, std=2e-5)

    return model
