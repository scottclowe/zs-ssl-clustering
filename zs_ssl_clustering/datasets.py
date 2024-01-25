"""
Handlers for various image datasets.
"""

import os
import socket
import warnings

import numpy as np
import sklearn.model_selection
import torch
import torchvision.datasets


def determine_host():
    r"""
    Determine which compute server we are on.

    Returns
    -------
    host : str, one of {"vaughan", "mars"}
        An identifier for the host compute system.
    """
    hostname = socket.gethostname()
    slurm_submit_host = os.environ.get("SLURM_SUBMIT_HOST")
    slurm_cluster_name = os.environ.get("SLURM_CLUSTER_NAME")

    if slurm_cluster_name and slurm_cluster_name.startswith("vaughan"):
        return "vaughan"
    if slurm_submit_host and slurm_submit_host in ["q.vector.local", "m.vector.local"]:
        return "mars"
    if hostname and hostname in ["q.vector.local", "m.vector.local"]:
        return "mars"
    if hostname and hostname.startswith("v"):
        return "vaughan"
    if slurm_submit_host and slurm_submit_host.startswith("v"):
        return "vaughan"
    if hostname and "srv.aau.dk" in hostname:
        return "aau"
    return ""


def image_dataset_sizes(dataset):
    r"""
    Get the image size and number of classes for a dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    Returns
    -------
    num_classes : int
        Number of classes in the dataset.
    img_size : int or None
        Size of the images in the dataset, or None if the images are not all
        the same size. Images are assumed to be square.
    num_channels : int
        Number of colour channels in the images in the dataset. This will be
        1 for greyscale images, and 3 for colour images.
    """
    dataset = dataset.lower().replace("-", "").replace("_", "").replace(" ", "")

    if dataset == "cifar10":
        num_classes = 10
        img_size = 32
        num_channels = 3

    elif dataset == "cifar100":
        num_classes = 100
        img_size = 32
        num_channels = 3

    elif dataset in ["imagenet", "imagenet1k", "ilsvrc2012"] or "imagenetv2" in dataset:
        num_classes = 1000
        img_size = None
        num_channels = 3

    elif dataset == "imageneto":
        num_classes = 200
        img_size = None
        num_channels = 3

    elif dataset in ["imagenetr", "imagenetrendition"]:
        num_classes = 200
        img_size = None
        num_channels = 3

    elif dataset == "imagenetsketch":
        num_classes = 1000
        img_size = None
        num_channels = 3

    elif dataset.startswith("imagenette"):
        num_classes = 10
        img_size = None
        num_channels = 3

    elif dataset.startswith("imagewoof"):
        num_classes = 10
        img_size = None
        num_channels = 3

    elif dataset.startswith("in9"):
        num_classes = 9
        img_size = None
        num_channels = 3

    elif dataset == "mnist":
        num_classes = 10
        img_size = 28
        num_channels = 1

    elif dataset == "fashionmnist":
        num_classes = 10
        img_size = 28
        num_channels = 1

    elif dataset == "kmnist":
        num_classes = 10
        img_size = 28
        num_channels = 1

    elif dataset == "svhn":
        num_classes = 10
        img_size = 32
        num_channels = 3

    elif dataset == "nabirds":
        num_classes = 555
        img_size = None
        num_channels = 3

    elif dataset in ["oxfordflowers102", "flowers102"]:
        num_classes = 102
        img_size = None
        num_channels = 3

    elif dataset == "stanfordcars":
        num_classes = 196
        img_size = None
        num_channels = 3

    elif dataset in ["inaturalist", "inaturalist-mini"]:
        num_classes = 10000
        img_size = None
        num_channels = 3

    elif dataset == "aircraft":
        num_classes = 100
        img_size = None
        num_channels = 3

    elif dataset == "celeba":
        num_classes = (
            10178  # There's actually 10177 classes, but the 0th class is unused
        )
        img_size = None
        num_channels = 3

    elif dataset == "utkface":
        num_classes = 117
        # There's actually fewer classes, but some ages don't appear
        img_size = 200
        num_channels = 3

    elif dataset == "dtd":
        num_classes = 47
        img_size = None
        num_channels = 3

    elif dataset == "lsun":
        num_classes = 10
        img_size = None
        num_channels = 3

    elif dataset == "places365":
        num_classes = 365
        img_size = None
        num_channels = 3

    elif dataset == "eurosat":
        num_classes = 10
        img_size = 64
        num_channels = 3

    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    return num_classes, img_size, num_channels


def fetch_image_dataset(
    dataset,
    root=None,
    transform_train=None,
    transform_eval=None,
    download=False,
):
    r"""
    Fetch a train and test dataset object for a given image dataset name.

    Parameters
    ----------
    dataset : str
        Name of dataset.
    root : str, optional
        Path to root directory containing the dataset.
    transform_train : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the training dataset.
    transform_eval : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the evaluation dataset.
    download : bool, optional
        Whether to download the dataset to the expected directory if it is not
        there. Only supported by some datasets. Default is ``False``.
    """
    dataset = dataset.lower().replace("-", "").replace("_", "").replace(" ", "")
    host = determine_host()

    if dataset == "cifar10":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd002/datasets/"
        elif host == "mars":
            root = "/scratch/gobi1/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.CIFAR10(
            os.path.join(root, dataset),
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.CIFAR10(
            os.path.join(root, dataset),
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset == "cifar100":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd002/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.CIFAR100(
            os.path.join(root, dataset),
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.CIFAR100(
            os.path.join(root, dataset),
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset in ["imagenet", "imagenet1k", "ilsvrc2012"]:
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        elif host == "mars":
            root = "/scratch/gobi1/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "imagenet", "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "imagenet", "val"),
            transform=transform_eval,
        )

    elif dataset in ["imagenetv2", "imagenetv2matchedfrequency"]:
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(
            root, "imagenetv2", "imagenetv2-matched-frequency-format-val"
        )
        dataset_train = None
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(root, transform=transform_eval)

    elif dataset == "imageneto":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(root, "imagenet-o")
        dataset_train = None
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(root, transform=transform_eval)

    elif dataset in ["imagenetr", "imagenetrendition"]:
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd002/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(root, "imagenet-r")
        dataset_train = None
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(root, transform=transform_eval)

    elif dataset == "imagenetsketch":
        # Manually download the data from links here:
        # https://github.com/HaohanWang/ImageNet-Sketch#download-the-data
        if root:
            pass
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(root, "imagenet-sketch")
        dataset_train = None
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(root, transform=transform_eval)

    elif dataset == "imagenette":
        if root:
            root = os.path.join(root, "imagenette")
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/imagenette2/full/"
        else:
            root = "~/Datasets/imagenette/"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "val"),
            transform=transform_eval,
        )

    elif dataset == "imagewoof":
        if root:
            root = os.path.join(root, "imagewoof")
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/imagewoof2/full/"
        else:
            root = "~/Datasets/imagewoof/"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "val"),
            transform=transform_eval,
        )

    elif dataset.startswith("in9"):
        if not root:
            root = "~/Datasets/in9_bg_challenge/"
        root = os.path.expanduser(root)
        if dataset in ["in9", "in9original"]:
            root = os.path.join(root, "original")
        elif dataset in ["in9l", "in9large"]:
            root = os.path.join(root, "in9l")
        elif dataset == "in9mixednext":
            root = os.path.join(root, "mixed_next")
        elif dataset == "in9mixedrand":
            root = os.path.join(root, "mixed_rand")
        elif dataset == "in9mixedsame":
            root = os.path.join(root, "mixed_same")
        elif dataset == "in9nofg":
            root = os.path.join(root, "no_fg")
        elif dataset == "in9onlybgb":
            root = os.path.join(root, "only_bg_b")
        elif dataset == "in9onlybgt":
            root = os.path.join(root, "only_bg_t")
        elif dataset == "in9onlyfg":
            root = os.path.join(root, "only_fg")
        else:
            raise ValueError(
                f"Unrecognised imagenet-9 backgrounds challenge dataset: {dataset}"
            )
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, "val"),
            transform=transform_eval,
        )

    elif dataset == "mnist":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/MNIST/processed
        dataset_train = torchvision.datasets.MNIST(
            root,
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.MNIST(
            root,
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset == "fashionmnist":
        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/FashionMNIST/raw
        dataset_train = torchvision.datasets.FashionMNIST(
            root,
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.FashionMNIST(
            root,
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset == "kmnist":
        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/KMNIST/raw
        dataset_train = torchvision.datasets.KMNIST(
            root,
            train=True,
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.KMNIST(
            root,
            train=False,
            transform=transform_eval,
            download=download,
        )

    elif dataset == "svhn":
        # SVHN has:
        #  73,257 digits for training,
        #  26,032 digits for testing, and
        # 531,131 additional, less difficult, samples to use as extra training data
        # We don't use the extra split here, only train. There are original
        # images which are large and have bounding boxes, but the pytorch class
        # just uses the 32px cropped individual digits.
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/"
        elif host == "mars":
            root = "/scratch/gobi1/datasets/"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.SVHN(
            os.path.join(root, "svhn"),
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.SVHN(
            os.path.join(root, "svhn"),
            split="test",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "nabirds":
        from zs_ssl_clustering.dataloaders.nabirds import NABirds

        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/NABirds/nabirds"
        else:
            root = "~/Datasets/nabirds"
        root = os.path.expanduser(root)
        dataset_train = NABirds(
            root,
            train=True,
            transform=transform_train,
            download=download if download else None,
        )
        dataset_val = None
        dataset_test = NABirds(
            root,
            train=False,
            transform=transform_eval,
            download=download if download else None,
        )

    elif dataset in ["oxfordflowers102", "flowers102"]:
        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/flowers-102
        dataset_train = torchvision.datasets.Flowers102(
            root,
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.Flowers102(
            root,
            split="test",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "stanfordcars":
        # Noticed on 2023-05-16 that the official source
        # https://ai.stanford.edu/~jkrause/cars/car_dataset.html
        # is not available
        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(root, dataset, "train"),
            transform=transform_train,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(root, dataset, "test"),
            transform=transform_eval,
        )

    elif dataset == "inaturalist":
        # Defaults to iNaturalist 2021 full train split
        # TODO Add older iNat versions?
        if root:
            pass
        else:
            root = "~/Datasets/iNaturalist"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.INaturalist(
            root,
            version="2021_train",
            target_type="full",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.INaturalist(
            root,
            version="2021_valid",
            target_type="full",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "inaturalistmini":
        # Defaults to iNaturalist 2021 mini train split
        # TODO Add older iNat versions?
        if root:
            pass
        else:
            root = "~/Datasets/iNaturalist"
        root = os.path.expanduser(root)
        dataset_train = torchvision.datasets.INaturalist(
            root,
            version="2021_mini",
            target_type="full",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.INaturalist(
            root,
            version="2021_valid",
            target_type="full",
            transform=transform_eval,
            download=download,
        )

    elif dataset in "aircraft":
        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/aircraft
        dataset_train = torchvision.datasets.FGVCAircraft(
            root,
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = torchvision.datasets.FGVCAircraft(
            root,
            split="test",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "celeba":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/ssd004/datasets/celeba_pytorch"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/celeba
        dataset_train = torchvision.datasets.CelebA(
            root,
            target_type="identity",
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = torchvision.datasets.CelebA(
            root,
            target_type="identity",
            split="valid",
            transform=transform_eval,
            download=download,
        )
        dataset_test = torchvision.datasets.CelebA(
            root,
            target_type="identity",
            split="test",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "utkface":
        from zs_ssl_clustering.dataloaders.utkface import UTKFace

        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/UTKFace
        dataset_train = UTKFace(
            root,
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = None
        dataset_test = UTKFace(
            root,
            split="test",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "dtd":
        if root:
            pass
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        # Will read from [root]/dtd
        dataset_train = torchvision.datasets.DTD(
            root,
            split="train",
            transform=transform_train,
            download=download,
        )
        dataset_val = torchvision.datasets.DTD(
            root,
            split="val",
            transform=transform_eval,
            download=download,
        )
        dataset_test = torchvision.datasets.DTD(
            root,
            split="test",
            transform=transform_eval,
            download=download,
        )

    elif dataset == "lsun":
        if root:
            pass
        elif host == "vaughan":
            root = "/scratch/hdd001/datasets/LSUN"
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(root, "lsun")
        dataset_train = torchvision.datasets.LSUN(
            root, classes="train", transform=transform_train
        )
        dataset_val = None
        dataset_test = torchvision.datasets.LSUN(
            root, classes="val", transform=transform_eval
        )

    elif dataset == "places365":
        if root:
            pass
        else:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(root, "Places365")
        # train-standard has 1.8M images
        # train-challenge has 8.0M images
        dataset_train = torchvision.datasets.Places365(
            root, split="train-standard", transform=transform_train
        )
        dataset_val = None
        dataset_test = torchvision.datasets.Places365(
            root, split="val", transform=transform_eval
        )

    elif dataset == "eurosat":
        # Download from
        # https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
        if not root:
            root = "~/Datasets"
        root = os.path.expanduser(root)
        root = os.path.join(root, "EuroSAT", "EuroSAT_RGB")
        dataset_train = torchvision.datasets.ImageFolder(
            root,
            transform=transform_train,
        )
        dataset_test = torchvision.datasets.ImageFolder(
            root,
            transform=transform_eval,
        )
        # Need to split the dataset to create a test set ourselves as it doesn't
        # come with any partitioning.
        dataset_train, dataset_test = create_train_val_split(
            dataset_train,
            dataset_test,
            split_rate=0.15,
            split_seed=0,
        )
        dataset_val = None

    else:
        raise ValueError("Unrecognised dataset: {}".format(dataset))

    return dataset_train, dataset_val, dataset_test


def fetch_dataset(
    dataset,
    root=None,
    prototyping=False,
    transform_train=None,
    transform_eval=None,
    protoval_split_rate="auto",
    protoval_split_seed=0,
    download=False,
):
    r"""
    Fetch a train and test dataset object for a given dataset name.

    Parameters
    ----------
    dataset : str
        Name of dataset.
    root : str, optional
        Path to root directory containing the dataset.
    prototyping : bool, default=False
        Whether to return a validation split distinct from the test split.
        If ``False``, the validation split will be the same as the test split
        for datasets which don't intrincally have a separate val and test
        partition.
        If ``True``, the validation partition is carved out of the train
        partition (resulting in a smaller training set) when there is no
        distinct validation partition available.
    transform_train : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the training dataset.
    transform_eval : callable, optional
        A function/transform that takes in an PIL image and returns a
        transformed version, to be applied to the evaluation dataset.
    protoval_split_rate : float or str, default="auto"
        The fraction of the train data to use for validating when in
        prototyping mode. If this is set to "auto", the split rate will be
        chosen such that the validation partition is the same size as the test
        partition.
    protoval_split_seed : int, default=0
        The seed to use for the split.
    download : bool, optional
        Whether to download the dataset to the expected directory if it is not
        there. Only supported by some datasets. Default is ``False``.

    Returns
    -------
    dataset_train : torch.utils.data.Dataset
        The training dataset.
    dataset_val : torch.utils.data.Dataset
        The validation dataset.
    dataset_test : torch.utils.data.Dataset
        The test dataset.
    distinct_val_test : bool
        Whether the validation and test partitions are distinct (True) or
        identical (False).
    """
    dataset_train, dataset_val, dataset_test = fetch_image_dataset(
        dataset=dataset,
        root=root,
        transform_train=transform_train,
        transform_eval=transform_eval,
        download=download,
    )

    # Handle the validation partition
    if dataset_val is not None:
        distinct_val_test = True
    elif not prototyping:
        dataset_val = dataset_test
        distinct_val_test = False
    else:
        # Create our own train/val split.
        #
        # For the validation part, we need a copy of dataset_train with the
        # evaluation transform instead.
        # The transform argument is *probably* going to be set to an attribute
        # on the dataset object called transform and called from there. But we
        # can't be completely sure, so to be completely agnostic about the
        # internals of the dataset class let's instantiate the dataset again!
        dataset_val = fetch_dataset(
            dataset,
            root=root,
            prototyping=False,
            transform_train=transform_eval,
        )[0]
        # dataset_val is a copy of the full training set, but with the transform
        # changed to transform_eval
        # Handle automatic validation partition sizing option.
        if not isinstance(protoval_split_rate, str):
            pass
        elif protoval_split_rate == "auto":
            # We want the validation set to be the same size as the test set.
            # This is the same as having a split rate of 1 - test_size.
            protoval_split_rate = len(dataset_test) / len(dataset_train)
        else:
            raise ValueError(f"Unsupported protoval_split_rate: {protoval_split_rate}")
        # Create the train/val split using these dataset objects.
        dataset_train, dataset_val = create_train_val_split(
            dataset_train,
            dataset_val,
            split_rate=protoval_split_rate,
            split_seed=protoval_split_seed,
        )
        distinct_val_test = True

    return (
        dataset_train,
        dataset_val,
        dataset_test,
        distinct_val_test,
    )


def create_train_val_split(
    dataset_train,
    dataset_val=None,
    split_rate=0.1,
    split_seed=0,
):
    r"""
    Create a train/val split of a dataset.

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        The full training dataset with training transforms.
    dataset_val : torch.utils.data.Dataset, optional
        The full training dataset with evaluation transforms.
        If this is not given, the source for the validation set will be
        ``dataset_test`` (with the same transforms as the training partition).
        Note that ``dataset_val`` must have the same samples as
        ``dataset_train``, and the samples must be in the same order.
    split_rate : float, default=0.1
        The fraction of the train data to use for the validation split.
    split_seed : int, default=0
        The seed to use for the split.

    Returns
    -------
    dataset_train : torch.utils.data.Dataset
        The training subset of the dataset.
    dataset_val : torch.utils.data.Dataset
        The validation subset of the dataset.
    """
    if dataset_val is None:
        dataset_val = dataset_train
    print(
        f"Creating prototyping train/val: {(1-split_rate)*100:.1f}/{split_rate*100:.1f}"
        f" with split seed={split_seed}."
    )
    # Try to do a stratified split.
    classes = get_dataset_labels(dataset_train)
    if classes is None:
        warnings.warn(
            "Creating prototyping splits without stratification.",
            UserWarning,
            stacklevel=2,
        )
    # Split the training set into train/val by sample index.
    train_indices, val_indices = sklearn.model_selection.train_test_split(
        np.arange(len(dataset_train)),
        test_size=split_rate,
        random_state=split_seed,
        stratify=classes,
    )
    # Create our splits. Assuming the dataset objects are always loaded
    # the same way, it will yield the same train/val split on each run.
    dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)
    return dataset_train, dataset_val


def get_dataset_labels(dataset):
    r"""
    Get the class labels within a :class:`torch.utils.data.Dataset` object.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset object.

    Returns
    -------
    array_like or None
        The class labels for each sample.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # For a dataset subset, we need to get the full set of labels from the
        # interior subset and then reduce them down to just the labels we have
        # in the subset.
        labels = get_dataset_labels(dataset.dataset)
        if labels is None:
            return labels
        return np.array(labels)[dataset.indices]

    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # For a concat dataset, we need to get the labels from each of the
        # interior datasets and then concatenate them together.
        labels = [get_dataset_labels(d) for d in dataset.datasets]
        for labels_i in labels:
            if labels_i is None:
                return None
        return np.concatenate(labels)

    labels = None
    if hasattr(dataset, "targets"):
        # MNIST, CIFAR, ImageFolder, etc
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        # STL10, SVHN
        labels = dataset.labels
    elif hasattr(dataset, "_labels"):
        # Flowers102
        labels = dataset._labels

    return labels
