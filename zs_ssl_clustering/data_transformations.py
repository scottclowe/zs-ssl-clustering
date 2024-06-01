import torch
from torchvision import transforms

NORMALIZATION = {
    "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    "clip": [(0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)],
    "mnist": [(0.1307,), (0.3081,)],
    "cifar": [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
}


def get_transform(
    zoom_ratio=1.0, image_size=224, image_channels=3, norm_type="imagenet"
):
    mean, std = NORMALIZATION[norm_type]

    steps = [
        transforms.Resize(int(image_size / zoom_ratio)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if image_channels == 1:
        # Convert greyscale image to have 3 channels
        steps.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    steps.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    transform = transforms.Compose(steps)
    return transform


def get_dna_transform(max_len=660, strip_trailing_n=True):
    steps = []
    if max_len is not None:
        steps.append(transforms.Lambda(lambda x: x[:max_len]))
    steps.append(transforms.Lambda(lambda x: x.rstrip("N") if strip_trailing_n else x))
    transform = transforms.Compose(steps)
    return transform
