import copy
import os
import re

import numpy as np


def sanitize_filename(text, allow_dotfiles=False):
    """
    Sanitise string so it can be a filename.

    Parameters
    ----------
    text : str
        A string.
    allow_dotfiles : bool, optional
        Whether to allow leading periods. Leading periods indicate hidden files
        on Linux. Default is `False`.

    Returns
    -------
    text : str
        The sanitised filename string.
    """
    # Remove non-ascii characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Folder names cannot end with a period in Windows. In Unix, a leading
    # period means the file or folder is normally hidden.
    # For this reason, we trim away leading and trailing periods as well as
    # spaces.
    if allow_dotfiles:
        text = text.strip().rstrip(".")
    else:
        text = text.strip(" .")

    # On Windows, a filename cannot contain any of the following characters:
    # \ / : * ? " < > |
    # Other operating systems are more permissive.
    # Replace / with a hyphen
    text = text.replace("/", "-")
    # Use a blacklist to remove any remaining forbidden characters
    text = re.sub(r'[\/:*?"<>|]+', "", text)
    return text


def get_embeddings_path(config, partition=None, modality=None):
    """
    Generate path to embeddings file.
    """
    if modality is None:
        modality = getattr(config, "modality", "image")
    extras = ""
    if modality != "image":
        extras += f"__{modality}"
    model = config.model if modality == "image" else config.dna_model
    fname = config.dataset_name + "__" + model + extras + ".npz"
    fname = sanitize_filename(fname)
    subdir = partition if partition is not None else config.partition
    if not isinstance(subdir, str):
        raise ValueError("Partition must be a string to load its embeddings.")
    if modality == "image":
        subdir += f"__z{config.zoom_ratio}"
    fname = os.path.join(config.embedding_dir, sanitize_filename(subdir), fname)
    return fname


def get_pred_path(config):
    """
    Generate path to y_pred file.
    """
    fname = f"{config.partition}-{config.dataset_name}"
    if "image" in config.modality:
        fname += f"__{config.model}"
    if "dna" in config.modality:
        fname += f"__{config.dna_model}"
    fname += f"__{config.run_id}.npz"
    fname = sanitize_filename(fname)
    fname = os.path.join(
        getattr(config, "predictions_dir", "y_pred"),
        sanitize_filename(config.partition),
        fname,
    )
    return fname


def load_embeddings(config, partitions=None, modalities=None):
    """
    Load cached embeddings.
    """
    if partitions is None:
        partitions = config.partition
    if modalities is None:
        modalities = getattr(config, "modality", "image")

    if not isinstance(partitions, str):
        embeddings = []
        y_true = []
        for partition in partitions:
            embeddings_i, y_true_i = load_embeddings(config, partition)
            embeddings.append(embeddings_i)
            y_true.append(y_true_i)
        # Concatenate the embeddings from each partition in the sample dimension
        embeddings = np.concatenate(embeddings, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        return embeddings, y_true

    partition = partitions

    if not isinstance(modalities, str):
        embeddings = []
        y_true = []
        for modality in modalities:
            embd_i, y_true_i = load_embeddings(config, partition, modality)
            embeddings.append(embd_i)
            y_true.append(y_true_i)
        # Concatenate the embeddings from each modality in the feature dimension
        embeddings = np.concatenate(embeddings, axis=1)
        for i in range(1, len(y_true)):
            assert np.all(y_true[i] == y_true[0]), "Mismatch in y_true labels"
        y_true = y_true[0]
        return embeddings, y_true

    modality = modalities

    if config.model in {"none", "raw"}:
        print("Using raw image pixel data instead of model embedding.", flush=True)
        from torch import nn

        import zs_ssl_clustering.embed

        _config = copy.deepcopy(config)
        _config.modality = modality
        dataloader = zs_ssl_clustering.embed.make_dataloader(_config)
        return zs_ssl_clustering.embed.embed_dataset(dataloader, nn.Flatten(), "cpu")

    fname = get_embeddings_path(config, partition, modality)
    print(f"Loading encoder embeddings from {fname}", flush=True)
    # Only need allow_pickle=True if we're using the saved config dict
    data = np.load(fname)
    return data["embeddings"], data["y_true"]
