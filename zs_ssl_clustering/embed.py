#!/usr/bin/env python

import builtins
import os
import time
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data

from zs_ssl_clustering import data_transformations, datasets, encoders, io, utils


def run(config):
    r"""
    Begin running the experiment on a single worker.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    config.batch_size_per_gpu = config.batch_size

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    n_cpus = utils.get_num_cpu_available()
    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {n_cpus} CPUs.")

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if not use_cuda:
        device = torch.device("cpu")
    elif config.gpu is None:
        device = "cuda"
    else:
        device = "cuda:{}".format(config.gpu)

    print(f"Using device {device}")

    # MODEL ===================================================================

    # Encoder -----------------------------------------------------------------
    # Build our Encoder.
    encoder = encoders.get_encoder(config.model)

    # Configure model for distributed training --------------------------------
    print("\nEncoder architecture:")
    print(encoder)
    print()

    if config.workers is None:
        config.workers = n_cpus

    if not torch.cuda.is_available():
        print("Using CPU (this will be slow)")
    else:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
        encoder = encoder.to(device)

    # DATASET =================================================================
    dataloader = make_dataloader(config, use_cuda=use_cuda)

    # Print configuration -----------------------------------------------------
    print()
    print("Configuration:")
    print()
    print(config)
    print()

    # EMBED ===================================================================
    # Ensure encoder is on the correct device
    encoder = encoder.to(device)
    # Create embeddings
    t0 = time.time()
    print("Creating embeddings...")
    embeddings, y_true = embed_dataset(dataloader, encoder, device)
    print(f"Created {len(embeddings)} embeddings in {time.time() - t0:.2f}s")

    # Save --------------------------------------------------------------------
    fname = io.get_embeddings_path(config)
    # Save embeddings
    if config.gpu_rank == 0:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        print(f"Saving embeddings to {fname}")
        t1 = time.time()
        tmp_a, tmp_b = os.path.split(fname)
        tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
        np.savez_compressed(
            tmp_fname,
            config=config,
            embeddings=embeddings,
            y_true=y_true,
        )
        os.rename(tmp_fname, fname)
        print(f"Saved embeddings in {time.time() - t1:.2f}s")


def make_dataloader(config, use_cuda=False):
    r"""
    Create a dataloader for a given partition of a dataset.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    use_cuda : bool, default=False
        Whether to use CUDA.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset.
    """
    n_class, raw_img_size, img_channels = datasets.image_dataset_sizes(
        config.dataset_name
    )

    if getattr(config, "image_size", None) is None:
        config.image_size = 224

    # Transforms --------------------------------------------------------------
    transform_eval = data_transformations.get_transform(
        getattr(config, "zoom_ratio", 1.0),
        image_size=config.image_size,
        image_channels=img_channels,
        norm_type="clip" if config.model.startswith("clip") else "imagenet",
    )

    # Dataset -----------------------------------------------------------------
    dataset_args = {
        "dataset": config.dataset_name,
        "root": getattr(config, "data_dir", None),
        "download": getattr(config, "allow_download_dataset", False),
    }
    if config.partition == "val":
        dataset_args["prototyping"] = True
    (
        dataset_train,
        dataset_val,
        dataset_test,
        distinct_val_test,
    ) = datasets.fetch_dataset(
        **dataset_args,
        transform_train=transform_eval,
        transform_eval=transform_eval,
    )
    if config.partition == "train":
        dataset = dataset_train
    elif config.partition == "val":
        dataset = dataset_val
    elif config.partition == "test":
        dataset = dataset_test
    elif config.partition == "train+test":
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    elif config.partition == "all":
        if distinct_val_test:
            datalist = [dataset_train, dataset_val, dataset_test]
        else:
            datalist = [dataset_train, dataset_test]
        dataset = torch.utils.data.ConcatDataset(datalist)
    else:
        raise ValueError(f"Unrecognized partition name: {config.partition}")

    # Dataloader --------------------------------------------------------------
    dl_kwargs = {
        "batch_size": getattr(config, "batch_size_per_gpu", 128),
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if use_cuda:
        cuda_kwargs = {"num_workers": config.workers, "pin_memory": True}
        dl_kwargs.update(cuda_kwargs)

    dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

    return dataloader


def embed_dataset(dataloader, encoder, device, is_distributed=False, log_interval=20):
    r"""
    Embed a dataset using the given encoder.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the dataset to embed.
    encoder : torch.nn.Module
        The encoder.
    device : torch.device
        Device to run the model on.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.
    log_interval : int, default=20
        How often to print progress creating the embeddings.

    Returns
    -------
    embeddings : np.ndarray
        The embeddings for each sample in the dataset.
    y_true : np.ndarray
        The class index for each sample in the dataset.
    """
    encoder.eval()
    embeddings_list = []
    y_true_list = []
    for i_batch, (stimuli, y_true) in enumerate(dataloader):
        if i_batch % log_interval == 0:
            print(f"Processing batch {i_batch + 1:3d}/{len(dataloader):3d}", flush=True)
        stimuli = stimuli.to(device)
        y_true = y_true.to(device)
        with torch.no_grad():
            embd = encoder(stimuli)
        if is_distributed:
            # Fetch results from other GPUs
            embd = utils.concat_all_gather_ragged(embd)
        if i_batch == 0:
            print(embd.shape)
        embeddings_list.append(embd.cpu().numpy())
        y_true_list.append(y_true.cpu().numpy())

    # Concatenate the embeddings and targets from each batch
    embeddings = np.concatenate(embeddings_list)
    y_true = np.concatenate(y_true_list)
    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    embeddings = embeddings[:n_samples]
    y_true = y_true[:n_samples]

    return embeddings, y_true


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train image classification model.",
        add_help=False,
    )
    # Help arg ----------------------------------------------------------------
    group = parser.add_argument_group("Help")
    group.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    # Dataset args ------------------------------------------------------------
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="cifar10",
        help="Name of the dataset to learn. Default: %(default)s",
    )
    group.add_argument(
        "--partition",
        type=str,
        default="test",
        help="Which partition of the dataset to use. Default: %(default)s",
    )
    group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory within which the dataset can be found."
            " Default is ~/Datasets, except on Vector servers where it is"
            " adjusted as appropriate depending on the dataset's location."
        ),
    )
    group.add_argument(
        "--allow-download-dataset",
        action="store_true",
        help="Attempt to download the dataset if it is not found locally.",
    )
    group.add_argument(
        "--zoom-ratio",
        type=float,
        default=1.0,
        help="Ratio of how much of the image to zoom in on. Default: %(default)s",
    )
    group.add_argument(
        "--image-size",
        type=int,
        help="Size of images to use as model input. Default: 224.",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        dest="model",
        type=str,
        default="resnet50",
        help="Name of model architecture. Default: %(default)s",
    )
    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--embedding-dir",
        type=str,
        metavar="PATH",
        default="embeddings",
        help="Path to directory containing embeddings.",
    )
    # Reproducibility args ----------------------------------------------------
    group = parser.add_argument_group("Reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random number generator (RNG) seed. Default: %(default)s",
    )
    mx_group = group.add_mutually_exclusive_group()
    mx_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Only use deterministic cuDNN features (disabled by default).",
    )
    mx_group.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Enable non-deterministic features of cuDNN.",
    )
    # Hardware configuration args ---------------------------------------------
    group = parser.add_argument_group("Hardware configuration")
    group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size. Default: %(default)s",
    )
    group.add_argument(
        "--workers",
        type=int,
        help="Number of CPU workers per node. Default: number of CPU cores on node.",
    )
    group.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU only, no GPUs.",
    )
    group.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="Index of GPU to use.",
    )
    # Logging args ------------------------------------------------------------
    group = parser.add_argument_group("Debugging and logging")
    group.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log results with Weights & Biases https://wandb.ai",
    )
    group.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Overrides --log-wandb and ensures wandb is always disabled.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        help=(
            "The entity (organization) within which your wandb project is"
            ' located. By default, this will be your "default location" set on'
            " wandb at https://wandb.ai/settings"
        ),
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="zs-ssl-clustering",
        help="Name of project on wandb, where these runs will be saved. Default: %(default)s",
    )
    group.add_argument(
        "--wandb-tags",
        nargs="+",
        type=str,
        help="Tag(s) to add to wandb run. Multiple tags can be given, separated by spaces.",
    )
    group.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="Used to group wandb runs together, to run stats on them together.",
    )
    group.add_argument(
        "--run-name",
        type=str,
        help="Human-readable identifier for the model run or job. Used to name the run on wandb.",
    )
    group.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for the model run or job. Used as the run ID on wandb.",
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    # Handle unspecified wandb_tags: if None, set to empty list
    if config.wandb_tags is None:
        config.wandb_tags = []
    return run(config)


if __name__ == "__main__":
    cli()
