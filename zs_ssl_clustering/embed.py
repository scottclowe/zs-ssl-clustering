#!/usr/bin/env python

import builtins
import os
import time
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from zs_ssl_clustering import data_transformations, datasets, encoders, io, utils


def run(config):
    r"""
    Begin running the experiment.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node * config.node_count
    config.distributed = config.world_size > 1
    config.batch_size = config.batch_size_per_gpu * config.world_size

    if config.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # run_one_worker process function
        torch.multiprocessing.spawn(
            run_one_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config)
        )
    else:
        # Simply call main_worker function once
        run_one_worker(config.gpu, ngpus_per_node, config)


def run_one_worker(gpu, ngpus_per_node, config):
    r"""
    Run one worker in the distributed training process.

    Parameters
    ----------
    gpu : int
        The GPU index of this worker, relative to this node.
    ngpus_per_node : int
        The number of GPUs per node.
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    config.gpu = gpu

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)
    elif config.distributed and not config.model_output_dir and not config.models_dir:
        raise ValueError(
            "A seed or checkpoint file must be specified for distributed training"
            " so that each GPU-worker starts with the same initial weights."
        )

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Suppress printing if this is not the master process for the node
    if config.distributed and config.gpu != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    try:
        n_cpus = len(os.sched_getaffinity(0))
    except BaseException:
        n_cpus = "UNK"

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Node rank {config.node_rank}")
    print(f"Found {torch.cuda.device_count()} GPUs and {n_cpus} CPUs.")

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    # DISTRIBUTION ============================================================
    if config.distributed:
        # For multiprocessing distributed training, gpu rank needs to be
        # set to the global rank among all the processes.
        config.gpu_rank = config.node_rank * ngpus_per_node + gpu
        print(f"GPU rank {config.gpu_rank} of {config.world_size}")
        print(
            f"Communicating with master worker {config.dist_url} via {config.dist_backend}"
        )
        torch.distributed.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.gpu_rank,
        )
        torch.distributed.barrier()
    else:
        config.gpu_rank = 0

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
    # We have to build the encoder before we load the dataset because it will
    # inform us about what size images we should produce in the preprocessing pipeline.
    encoder = encoders.get_encoder(config.model)

    n_class, raw_img_size, img_channels = datasets.image_dataset_sizes(
        config.dataset_name
    )
    encoder_config = getattr(encoders, "data_config", {})

    if config.image_size is None:
        if "input_size" in encoder_config:
            config.image_size = encoder_config["input_size"][-1]
            print(
                f"Setting model input image size to encoder's expected input size: {config.image_size}"
            )
        else:
            config.image_size = 224
            print(f"Setting model input image size to default: {config.image_size}")
            if raw_img_size:
                warnings.warn(
                    "Be aware that we are using a different input image size"
                    f" ({config.image_size}px) to the raw image size in the"
                    f" dataset ({raw_img_size}px).",
                    UserWarning,
                    stacklevel=2,
                )
    elif (
        "input_size" in encoder_config
        and encoder_config["input_size"][-1] != config.image_size
    ):
        warnings.warn(
            f"A different image size {config.image_size} than what the model was"
            f" pretrained with {encoder_config['input_size'][-1]} was suplied",
            UserWarning,
            stacklevel=2,
        )

    # Configure model for distributed training --------------------------------
    print("\nEncoder architecture:")
    print(encoder)
    print()

    if config.workers is None:
        if n_cpus != "UNK":
            config.workers = n_cpus
        else:
            raise ValueError("Could not read the number of available CPUs")

    if not torch.cuda.is_available():
        print("Using CPU (this will be slow)")
    elif config.distributed:
        # For multiprocessing distributed, the DistributedDataParallel
        # constructor should always set a single device scope, otherwise
        # DistributedDataParallel will use all available devices.
        encoder.to(device)
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            encoder = nn.parallel.DistributedDataParallel(
                encoder, device_ids=[config.gpu]
            )
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            encoder = nn.parallel.DistributedDataParallel(encoder)
    else:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
        encoder = encoder.to(device)

    # DATASET =================================================================
    # Transforms --------------------------------------------------------------
    transform_eval = data_transformations.get_transform(
        config.zoom_ratio,
        image_size=config.image_size,
        image_channels=img_channels,
    )

    # Dataset -----------------------------------------------------------------
    dataset_args = {
        "dataset": config.dataset_name,
        "root": config.data_dir,
        "download": config.allow_download_dataset,
    }
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
        dataset = torch.utils.data.ConcatDataset(dataset_train, dataset_test)
    else:
        raise ValueError(f"Unrecognized partition name: {config.partition}")

    # Dataloader --------------------------------------------------------------
    dl_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if use_cuda:
        cuda_kwargs = {"num_workers": config.workers, "pin_memory": True}
        dl_kwargs.update(cuda_kwargs)

    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_kwargs["sampler"] = DistributedSampler(
            dataset,
            shuffle=False,
            drop_last=False,
        )
        dl_kwargs["shuffle"] = None

    dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

    # TRAIN ===================================================================
    print()
    print("Configuration:")
    print()
    print(config)
    print()

    # Ensure encoder is on the correct device
    encoder = encoder.to(device)
    # Create embeddings
    t0 = time.time()
    print("Creating embeddings...")
    embeddings, y_true = embed_dataset(
        dataloader, encoder, device, is_distributed=config.distributed
    )
    print(f"Created {len(embeddings)} embeddings in {time.time() - t0:.2f}s")
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


def embed_dataset(dataloader, encoder, device, is_distributed=False):
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
    for stimuli, y_true in dataloader:
        stimuli = stimuli.to(device)
        y_true = y_true.to(device)
        with torch.no_grad():
            embd = encoder(stimuli)
        if is_distributed:
            # Fetch results from other GPUs
            embd = utils.concat_all_gather_ragged(embd)
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
        help="Size of images to use as model input. Default: encoder's default.",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        dest="model",
        type=str,
        default="resnet18",
        help="Name of model architecture. Default: %(default)s",
    )
    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--output-dir",
        type=str,
        metavar="PATH",
        default="embeddings",
        help="Path to output directory.",
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
        dest="batch_size_per_gpu",
        type=int,
        default=128,
        help=(
            "Batch size per GPU. The total batch size will be this value times"
            " the total number of GPUs used. Default: %(default)s"
        ),
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
        help="Index of GPU to use. Setting this will disable GPU parallelism.",
    )
    group.add_argument(
        "--node-count",
        default=1,
        type=int,
        help="Number of nodes for distributed training.",
    )
    group.add_argument(
        "--node-rank",
        "--rank",
        dest="node_rank",
        default=0,
        type=int,
        help="Node rank for distributed training.",
    )
    group.add_argument(
        "--dist-url",
        default="tcp://localhost:23456",
        type=str,
        help="URL used to set up distributed training.",
    )
    group.add_argument(
        "--dist-backend",
        default="nccl",
        type=str,
        help=(
            "Distributed training backend. Must be supported by"
            " torch.distributed (one of gloo, mpi, nccl), and supported by"
            " your GPU server."
        ),
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
