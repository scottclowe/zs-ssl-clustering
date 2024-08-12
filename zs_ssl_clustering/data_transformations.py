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


class RandomRotate90(torch.nn.Module):
    """
    Randomly rotate the input tensor by 0, 90, 180, or 270 degrees.

    Assumes input tensor is of shape (..., H, W).
    """

    def __init__(self, p=0.75):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return x
        k = torch.randint(1, 4, (1,)).item()
        return torch.rot90(x, k, dims=[-2, -1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_randsizecrop_transform(
    image_size=224,
    image_channels=3,
    norm_type="imagenet",
    ratio=None,
    hflip=0,
    rotate=False,
):
    mean, std = NORMALIZATION[norm_type]
    if ratio is None:
        ratio = (0.75, 1.3333333333333333)
    elif ratio == 0:
        ratio = (1, 1)
    elif isinstance(ratio, (int, float)):
        ratio = (ratio, 1 / ratio)

    steps = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=ratio),
        transforms.ToTensor(),
    ]
    if hflip:
        if hflip == 1:
            hflip = 0.5
        steps.append(transforms.RandomHorizontalFlip(p=hflip))
    if rotate:
        steps.append(RandomRotate90())
    if image_channels == 1:
        # Convert greyscale image to have 3 channels
        steps.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    steps.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    transform = transforms.Compose(steps)
    return transform
