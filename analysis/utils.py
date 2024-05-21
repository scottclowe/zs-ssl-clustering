import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import wandb


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


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Create a colormap with a certain number of shades of colours.

    https://stackoverflow.com/a/47232942/1960959
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc : (i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


def clip_imgsize(dataset, target_image_size):
    if target_image_size is None:
        return target_image_size
    dataset_imsize = image_dataset_sizes(dataset)[1]
    if dataset_imsize is None:
        return target_image_size
    return min(target_image_size, dataset_imsize)


def fixup_filter(filters):
    dataset = filters.get("dataset_name", filters.get("dataset", None))
    if dataset and "image_size" in filters:
        filters["image_size"] = clip_imgsize(dataset, filters["image_size"])
    if dataset and "min_samples" in filters:
        if dataset.lower() in ["celeba", "utkface", "bioscan1m"]:
            filters["min_samples"] = 2
    return filters


def select_rows(df, filters, allow_missing=True, fixup=True):
    if fixup:
        filters = fixup_filter(filters)
    select = np.ones(len(df), dtype=bool)
    for col, val in filters.items():
        if col == "dataset":
            col = "dataset_name"
        if col == "clusterer":
            col = "clusterer_name"
        if val is None or val == "None" or val == "none":
            select_i = pd.isna(df[col])
            select_i |= df[col] == "None"
            select_i |= df[col] == "none"
        else:
            select_i = df[col] == val
            select_i |= df[col] == str(val)
            if allow_missing or val == "None" or val == "none":
                select_i |= pd.isna(df[col])
        select &= select_i
    return df[select]


def find_differing_columns(df, cols=None):
    if cols is None:
        cols = df.columns
    my_cols = []
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=False) > 1:
            my_cols.append(col)
    return my_cols


def filter2command(*filters, partition="val"):
    f = {}
    for filter in filters:
        for k, v in filter.items():
            f[k] = v
    dataset = f.get("dataset", "")
    clusterer = f.get("clusterer", "")

    mem = 2  # RAM in gigabytes

    if clusterer in ["LouvainCommunities"]:
        if dataset in ["inaturalist"]:
            # 100,000 samples
            mem = 3_700
        elif dataset in ["imagenet-sketch", "imagenet"]:
            # 50,000 samples
            mem = 926
        elif dataset in ["places365"]:
            # 36,500 samples
            mem = 494
        elif dataset in ["imagenet-r"]:
            # 30,000 samples
            mem = 333
        elif dataset in ["svhn"]:
            # 26,000 samples
            mem = 250
        elif dataset in ["bioscan1m", "nabirds"]:
            # 24,600 samples
            mem = 224
        elif dataset in ["celeba"]:
            # 20,000 samples
            mem = 128
        elif dataset in [
            "imagenetv2",
            "cifar10",
            "cifar100",
            "lsun",
            "mnist",
            "fashionmnist",
            "stanfordcars",
            "breakhis",
        ]:
            # 8,000 - 10,000 samples
            mem = 32
        elif dataset in ["flowers102", "utkface"]:
            # 5,925 - 6,200 samples
            mem = 18
        elif dataset.startswith("in9") or dataset in ["eurosat"]:
            # 4,500 samples
            mem = 8
        elif dataset in ["imagenette", "imagewoof", "aircraft"]:
            # 3,333 - 3,930 samples
            mem = 6
        elif dataset in ["imagenet-o", "dtd"]:
            # 2,000 samples
            mem = 4
        else:
            mem = 12

    elif clusterer in ["AffinityPropagation"]:
        if dataset in ["inaturalist"]:
            # 100,000 samples
            mem = 292
        elif dataset in ["imagenet-sketch", "imagenet"]:
            # 50,000 samples
            mem = 72
        elif dataset in ["places365", "imagenet-r", "svhn", "bioscan1m", "nabirds"]:
            # 24,600 - 36,500 samples
            mem = 48
        elif dataset in ["celeba"]:
            # 20,000 samples
            mem = 12
        elif dataset in [
            "imagenetv2",
            "cifar10",
            "cifar100",
            "lsun",
            "mnist",
            "fashionmnist",
            "stanfordcars",
        ]:
            # 8,000 - 10,000 samples
            mem = 6
        elif dataset.startswith("in9") or dataset in [
            "flowers102",
            "utkface",
            "eurosat",
            "aircraft",
            "breakhis",
            "imagenet-o",
            "dtd",
        ]:
            # 1,900 - 6,200 samples
            mem = 2
        elif dataset in ["imagenette", "imagewoof"]:
            # 3,930 samples
            mem = 1
        else:
            mem = 8

    elif clusterer in ["AgglomerativeClustering", "SpectralClustering"]:
        if dataset in ["inaturalist"]:
            # 100,000 samples
            mem = 72
        elif dataset in ["imagenet-sketch", "imagenet"]:
            # 50,000 samples
            mem = 20
        elif dataset in ["places365", "imagenet-r", "svhn", "bioscan1m", "nabirds"]:
            # 24,600 - 36,500 samples
            mem = 16
        elif dataset in ["celeba"]:
            # 20,000 samples
            mem = 12
        elif dataset in [
            "imagenetv2",
            "cifar10",
            "cifar100",
            "lsun",
            "mnist",
            "fashionmnist",
            "stanfordcars",
        ]:
            # 8,000 - 10,000 samples
            mem = 6
        elif dataset.startswith("in9") or dataset in [
            "flowers102",
            "utkface",
            "eurosat",
            "aircraft",
            "breakhis",
            "imagenet-o",
            "dtd",
        ]:
            # 1,900 - 6,200 samples
            mem = 4
        elif dataset in ["imagenette", "imagewoof"]:
            # 3,930 samples
            mem = 2
        else:
            mem = 8
        if clusterer == "SpectralClustering":
            snn = f.get("spectral_n_neighbors", 100)
            if snn <= 10:
                mem = mem * 8 / 20
            elif snn <= 20:
                mem = mem * 3 / 4
            mem = int(np.ceil(mem))

    elif clusterer in ["HDBSCAN", "KMeans"]:
        if dataset in ["inaturalist"]:
            # 100,000 samples
            mem = 6
        elif dataset in ["imagenet-sketch", "imagenet"]:
            # 50,000 samples
            mem = 4
        elif dataset in ["places365", "imagenet-r", "svhn", "bioscan1m", "nabirds"]:
            # 24,600 - 36,500 samples
            mem = 4
        elif dataset in ["celeba"]:
            # 20,000 samples
            mem = 4
        elif dataset in [
            "imagenetv2",
            "cifar10",
            "cifar100",
            "lsun",
            "mnist",
            "fashionmnist",
            "stanfordcars",
        ]:
            # 8,000 - 10,000 samples
            mem = 2
        elif dataset.startswith("in9") or dataset in [
            "flowers102",
            "utkface",
            "eurosat",
            "aircraft",
            "breakhis",
            "imagenet-o",
            "dtd",
        ]:
            # 1,900 - 6,200 samples
            mem = 2
        elif dataset in ["imagenette", "imagewoof"]:
            # 3,930 samples
            mem = 1
        else:
            mem = 4

    if mem > 300:
        return ""
    if mem > 129:
        pass

    mem = f"{mem}G"

    if partition == "val":
        seed = 100
    elif partition == "test":
        seed = 1
    else:
        seed = 0
    s = (
        f"sbatch --array={seed} --mem={mem}"
        f' --job-name="zsc-{f.get("model", "")}-{dataset}-{clusterer}"'
        f" slurm/cluster.slrm --partition={partition}"
    )
    for k, v in f.items():
        if v is None:
            continue
        if k == "zscore":
            if v == "False" or not v:
                s += " --no-zscore"
            elif v == "True" or v:
                s += " --zscore"
            continue
        if k == "normalize":
            if v == "False" or not v:
                pass
            elif v == "True" or v:
                s += " --normalize"
            continue
        if k == "zscore2":
            if v == "False" or not v:
                s += " --no-zscore2"
            elif v == "average":
                s += " --azscore2"
            elif v == "standard" or v:
                s += " --zscore2"
            continue
        if k == "ndim_correction":
            if v == "False" or not v:
                s += " --no-ndim-correction"
            elif v == "True" or v:
                s += " --ndim-correction"
            continue
        if k == "louvain_remove_self_loops":
            if v == "False" or not v:
                s += " --louvain-keep-self"
            elif v == "True" or v:
                pass
            continue
        s += f" --{k.replace('_', '-')}={v}"
    return s


def fetch_results():
    runs_df_long = pd.DataFrame({"id": []})
    config_keys = set()
    summary_keys = set()

    # Project is specified by <entity/project-name>
    api = wandb.Api(timeout=720)
    runs = api.runs(
        "uoguelph_mlrg/zs-ssl-clustering",
        filters={
            "state": "Finished",
            "config.partition": "test",
        },  # "config.predictions_dir": "y_pred"},
        per_page=10_000,
    )
    len(runs)

    print(f"{len(runs_df_long)} runs currently in dataframe")
    rows_to_add = []
    existing_ids = set(runs_df_long["id"].values)
    for run in runs:
        if run.id in existing_ids:
            if len(rows_to_add) >= len(runs) - len(runs_df_long):
                break
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # .name is the human-readable name of the run.
        row = {"id": run.id, "name": run.name}
        row.update({k: v for k, v in config.items() if not k.startswith("_")})
        row.update({k: v for k, v in summary.items() if not k.startswith("_")})
        if "_timestamp" in summary:
            row["_timestamp"] = summary["_timestamp"]
        rows_to_add.append(row)
        config_keys = config_keys.union(config.keys())
        summary_keys = summary_keys.union(summary.keys())

    if not len(rows_to_add):
        print("No new runs to add")
    else:
        print(f"Adding {len(rows_to_add)} runs")
        runs_df_long = pd.concat([runs_df_long, pd.DataFrame.from_records(rows_to_add)])
    print(f"{len(runs_df_long)} runs")

    # Remove entries without an AMI metric
    test_runs_df = runs_df_long[~runs_df_long["AMI"].isna()]
    len(test_runs_df)

    # Handle changed default value for spectral_assigner after config arg was introduced
    if "spectral_n_components" not in test_runs_df.columns:
        test_runs_df["spectral_n_components"] = None

    if "spectral_assigner" not in test_runs_df.columns:
        test_runs_df["spectral_assigner"] = None
    select = test_runs_df["clusterer_name"] != "SpectralClustering"
    test_runs_df.loc[select, "spectral_assigner"] = None
    select = (test_runs_df["clusterer_name"] == "SpectralClustering") & pd.isna(
        test_runs_df["spectral_assigner"]
    )
    test_runs_df.loc[select, "spectral_assigner"] = "kmeans"

    # Accidentally wasn't clearing this hparam when it was unused
    if "spectral_affinity" not in test_runs_df.columns:
        test_runs_df["spectral_affinity"] = None
    select = test_runs_df["clusterer_name"] != "SpectralClustering"
    test_runs_df.loc[select, "spectral_affinity"] = None

    if "zscore2" not in test_runs_df.columns:
        test_runs_df["zscore2"] = False
    test_runs_df.loc[pd.isna(test_runs_df["zscore2"]), "zscore2"] = False

    if "ndim_correction" not in test_runs_df.columns:
        test_runs_df["ndim_correction"] = False
    test_runs_df.loc[
        pd.isna(test_runs_df["ndim_correction"]), "ndim_correction"
    ] = False

    if "dim_reducer_man_nn" not in test_runs_df.columns:
        test_runs_df["dim_reducer_man_nn"] = None

    if "image_size" not in test_runs_df.columns:
        test_runs_df["image_size"] = None

    config_keys = config_keys.difference(
        {"workers", "memory_avail_GB", "memory_total_GB", "memory_slurm"}
    )

    return test_runs_df, config_keys


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


def get_pred_path(row):
    """
    Generate path to y_pred file.
    """
    run_id = row["name"].split("__")[-1]
    fname = f"{row['partition']}-{row['dataset_name']}__{row['model']}__{run_id}.npz"
    fname = sanitize_filename(fname)
    fname = os.path.join(
        row["predictions_dir"],
        sanitize_filename(row["partition"] + f"__z{float(row['zoom_ratio'])}"),
        fname,
    )
    return fname
