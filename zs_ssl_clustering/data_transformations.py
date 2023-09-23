import torch
from torchvision import transforms

NORMALIZATION = {
    "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    "mnist": [(0.1307,), (0.3081,)],
    "cifar": [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
}

VALID_TRANSFORMS = ["imagenet", "cifar", "mnist"]


def get_transform(zoom_ratio=1.0, image_size=224, args=None):
    if args is None:
        args = {}
    mean, std = NORMALIZATION[args.get("normalization", "imagenet")]
    if "mean" in args:
        mean = args["mean"]
    if "std" in args:
        std = args["std"]

    steps = [
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if zoom_ratio != 1.0:
        steps.insert(0, transforms.Resize(int(image_size / 0.875 * zoom_ratio)))
    transform = transforms.Compose(steps)
    return transform
