import copy

import matplotlib.colors

# DEFINITIONS
VALIDATION_DATASETS = ["imagenet", "imagenette", "imagewoof"]
RESNET50_MODELS = [
    "random_resnet50",
    "resnet50",
    "mocov3_resnet50",
    "dino_resnet50",
    "vicreg_resnet50",
    "clip_RN50",
]
VITB16_MODELS = [
    "random_vitb16",
    "vitb16",
    "mocov3_vit_base",
    "dino_vitb16",
    "timm_vit_base_patch16_224.mae",
    "mae_pretrain_vit_base_global",
    "clip_vitb16",
]

# # Exclude CLIP from analysis
# RESNET50_MODELS = [v for v in RESNET50_MODELS if not v.startswith("clip")]
# VITB16_MODELS = [v for v in VITB16_MODELS if not v.startswith("clip")]

FT_RESNET50_MODELS = [
    "ft_mocov3_resnet50",
    "ft_dino_resnet50",
    "ft_vicreg_resnet50",
]
FT_VITB16_MODELS = [
    "ft_mocov3_vit_base",
    "ft_dino_vitb16",
    "mae_finetuned_vit_base_global",
]
FT_MODELS = FT_RESNET50_MODELS + FT_VITB16_MODELS
ALL_MODELS = (
    ["none"] + RESNET50_MODELS + VITB16_MODELS + FT_RESNET50_MODELS + FT_VITB16_MODELS
)

RESNET50_MODELS_INTERLEAVED = [
    "random_resnet50",
    "resnet50",
    "mocov3_resnet50",
    "ft_mocov3_resnet50",
    "dino_resnet50",
    "ft_dino_resnet50",
    "vicreg_resnet50",
    "ft_vicreg_resnet50",
]
VITB16_MODELS_INTERLEAVED = [
    "random_vitb16",
    "vitb16",
    "mocov3_vit_base",
    "ft_mocov3_vit_base",
    "dino_vitb16",
    "ft_dino_vitb16",
    "timm_vit_base_patch16_224.mae",
    "mae_pretrain_vit_base_global",
    "mae_finetuned_vit_base_global",
]

CLUSTERERS = [
    "KMeans",
    "LouvainCommunities",
    "AgglomerativeClustering",
    "AffinityPropagation",
    "SpectralClustering",
    "HDBSCAN",
    "OPTICS",
]
ALL_CLUSTERERS = copy.deepcopy(CLUSTERERS)
DISTANCE_METRICS = [
    "euclidean",
    "l1",
    "chebyshev",
    "cosine",
    "arccos",
    "braycurtis",
    "canberra",
]

PRE2FT = {
    k: "ft_" + k
    for k in [
        "mocov3_resnet50",
        "dino_resnet50",
        "vicreg_resnet50",
        "mocov3_vit_base",
        "dino_vitb16",
    ]
}
PRE2FT["mae_pretrain_vit_base_global"] = "mae_finetuned_vit_base_global"
FT2PRE = {v: k for k, v in PRE2FT.items()}

DATASET2LS = {
    "imagenet": "-.",
    "imagenette": "--",
    "imagewoof": ":",
}


DEFAULT_PARAMS = {
    "all": {
        "dim_reducer": "None",
        "dim_reducer_man": "None",
        "zscore": False,
        "normalize": False,
        "zscore2": False,
        "ndim_correction": False,
    },
    "KMeans": {"clusterer": "KMeans"},
    "LouvainCommunities": {
        "clusterer": "LouvainCommunities",
        "louvain_resolution": 1.0,
        "louvain_threshold": 1e-7,
        "louvain_remove_self_loops": False,
        "distance_metric": "l2",
    },
    "AffinityPropagation": {
        "clusterer": "AffinityPropagation",
        "affinity_damping": 0.9,
        "affinity_conv_iter": 15,
    },
    "SpectralClustering": {
        "clusterer": "SpectralClustering",
        "spectral_assigner": "cluster_qr",
        "spectral_affinity": "nearest_neighbors",
        "spectral_n_neighbors": 10,
        "spectral_n_components": None,
    },
    "AgglomerativeClustering": {
        "clusterer": "AgglomerativeClustering",
        "distance_metric": "euclidean",
        "aggclust_linkage": "ward",
    },
    "HDBSCAN": {
        "clusterer": "HDBSCAN",
        "hdbscan_method": "eom",
        "min_samples": 5,
        "max_samples": 0.2,
        "distance_metric": "euclidean",
    },
    "OPTICS": {
        "clusterer": "OPTICS",
        "optics_method": "xi",
        "optics_xi": 0.05,
        "distance_metric": "euclidean",
    },
}

TEST_DATASETS = [
    "imagenet",
    "imagenetv2",
    "imagenet-o",
    "cifar10",
    "cifar100",
    "in9original",
    "in9mixedrand",
    # "in9onlybgt",
    "in9onlyfg",
    "imagenet-r",
    "imagenet-sketch",
    "aircraft",
    "stanfordcars",
    "flowers102",
    "bioscan1m",
    "nabirds",
    "inaturalist",
    "celeba",
    "utkface",
    "breakhis",
    "dtd",
    "eurosat",
    "lsun",
    "places365",
    "mnist",
    "fashionmnist",
    "svhn",
]
DATASET2SH = {
    "aircraft": "Air",
    "bioscan1m": "Bio",
    "breakhis": "BHis",
    "celeba": "CelA",
    "cifar10": "C10",
    "cifar100": "C100",
    "dtd": "DTD",
    "eurosat": "ESAT",
    "flowers102": "F102",
    "fashionmnist": "Fash",
    "imagenet": "IN1k",
    "imagenet-o": "IN-O",
    "imagenet-r": "IN-R",
    "imagenet-sketch": "IN-S",
    "imagenetv2": "INv2",
    "imagenette": "IN10",
    "imagewoof": "INwf",
    "in9original": "IN9",
    "in9mixednext": "9-MN",
    "in9mixedrand": "9-MR",
    "in9mixedsame": "9-MS",
    "in9nofg": "9-NoFG",
    "in9onlybgb": "9-BGB",
    "in9onlybgt": "9-BGT",
    "in9onlyfg": "9-FG",
    "inaturalist": "iNat",
    "lsun": "LSU",
    "mnist": "MNST",
    "nabirds": "Birds",
    "places365": "P365",
    "stanfordcars": "Cars",
    "svhn": "SVHN",
    "utkface": "UTKF",
}
MODEL_GROUPS = {
    "ResNet-50": RESNET50_MODELS,
    "ViT-B": VITB16_MODELS,
    "ResNet-50 [FT]": FT_RESNET50_MODELS,
    "ViT-B [FT]": FT_VITB16_MODELS,
    "all": ALL_MODELS,
}
MODEL2SH = {
    "none": "Raw image",
    "random_resnet50": "Rand.",  # "Random",
    "random_vitb16": "Rand.",  # "Random",
    "resnet50": "X-Ent.",
    "mocov3_resnet50": "MoCo-v3",
    "dino_resnet50": "DINO",
    "vicreg_resnet50": "VICReg",
    "clip_RN50": "CLIP",
    "vitb16": "X-Ent.",
    "mocov3_vit_base": "MoCo-v3",
    "dino_vitb16": "DINO",
    "timm_vit_base_patch16_224.mae": "MAE (CLS)",
    "mae_pretrain_vit_base_global": "MAE (avg)",
    "clip_vitb16": "CLIP",
    "ft_mocov3_resnet50": "MoCo-v3 [FT]",
    "ft_dino_resnet50": "DINO [FT]",
    "ft_vicreg_resnet50": "VICReg [FT]",
    "ft_mocov3_vit_base": "MoCo-v3 [FT]",
    "ft_dino_vitb16": "DINO [FT]",
    "mae_finetuned_vit_base_global": "MAE (avg) [FT]",
}
CLUSTERER2SH = {
    "KMeans": "K-Means",
    "SpectralClustering": "Spectral",
    "AffinityPropagation": "Affinity Prop",
    "AgglomerativeClustering": "AC",
    "AC w/ C": "AC w/  C",
}

MODEL2SH_ARCH = dict(MODEL2SH)
for k, v in MODEL2SH.items():
    if "resnet" in k or "RN50" in k:
        MODEL2SH_ARCH[k] = f"ResNet-50 {v}"
    elif "vit" in k:
        MODEL2SH_ARCH[k] = f"ViT-B {v}"

TEST_DATASETS_GROUPED = {
    "In-domain": [
        "imagenet",
        "imagenetv2",
        "cifar10",
        "cifar100",
        "in9original",
    ],
    "Domain-shift": [
        "in9onlyfg",
        # "in9onlybgt",
        "in9mixedrand",
        "imagenet-r",
        "imagenet-sketch",
    ],
    "Near-OOD": [
        "imagenet-o",
        "lsun",
        "places365",
    ],
    "Fine-grained": [
        "aircraft",
        "stanfordcars",
        "flowers102",
        "bioscan1m",
        "nabirds",
        "inaturalist",
    ],
    "Far-OOD": [
        "celeba",
        "utkface",
        "breakhis",
        "dtd",
        "eurosat",
        "mnist",
        "fashionmnist",
        "svhn",
    ],
}

DATASETGROUP2TITLE = {
    "Domain-shift": "Domain-shifted",
    "Out-of-distribution": "OOD",
}

IN9_DATASETS = [
    "in9original",
    "in9onlyfg",
    "in9nofg",
    "in9onlybgt",
    "in9mixedsame",
    "in9mixedrand",
]
IN92SH = {
    "in9original": "OG",
    "in9mixednext": "MN",
    "in9mixedrand": "MR",
    "in9mixedsame": "MS",
    "in9nofg": r"FG$^\text{C}$",
    "in9onlybgb": "BG(B)",
    "in9onlybgt": "BG",
    "in9onlyfg": "FG",
    "in9bggap": "Gap",
}
CLUSTERER2COLORSTR = {
    "KMeans": "tab:purple",
    "SpectralClustering": "tab:cyan",
    "AC w/ C": "tab:red",
    "AC w/o C": "tab:orange",
    "AffinityPropagation": "tab:green",
    "HDBSCAN": "tab:blue",
}
CLUSTERER2COLORRGB = {
    k: matplotlib.colors.to_rgb(v) for k, v in CLUSTERER2COLORSTR.items()
}

# ICML2024
MODEL2COLORSTR = {
    "none": "black",
    "random_resnet50": "dimgrey",
    "random_vitb16": "dimgrey",
    "resnet50": "tab:red",
    "mocov3_resnet50": "tab:green",
    "dino_resnet50": "tab:purple",
    "vicreg_resnet50": "tab:orange",
    "clip_RN50": "tab:olive",
    "vitb16": "tab:red",
    "mocov3_vit_base": "tab:green",
    "dino_vitb16": "tab:purple",
    "timm_vit_base_patch16_224.mae": "tab:blue",
    "mae_pretrain_vit_base_global": "tab:brown",
    "clip_vitb16": "tab:olive",
    "mae_finetuned_vit_base_global": "tab:brown",
}
MODEL2COLORRGB = {k: matplotlib.colors.to_rgb(v) for k, v in MODEL2COLORSTR.items()}
for model in FT_MODELS:
    MODEL2COLORRGB[model] = tuple(c * 0.8 for c in MODEL2COLORRGB[FT2PRE[model]])
for model in RESNET50_MODELS + VITB16_MODELS:
    MODEL2COLORRGB[model] = tuple(1 - (1 - c) * 0.7 for c in MODEL2COLORRGB[model])


def setup_best_params():
    models = RESNET50_MODELS + VITB16_MODELS
    BEST_PARAMS = {
        clusterer: {model: copy.deepcopy(DEFAULT_PARAMS[clusterer]) for model in models}
        for clusterer in ALL_CLUSTERERS
    }

    # KMeans
    # Use UMAP (num dims unimportant; we select 50d for consistency) for every encoder except
    # - clip_RN50 : a little better to use PCA with 500d than UMAP. UMAP beats PCA if you
    #   reduce the PCA dims below 500.
    # - clip_vitb16 : same behaviour as clip_RN50
    # - timm_vit_base_patch16_224.mae : best is PCA 0.85 variance explained. Need at least
    #   200 PCA dims, and PCA perf beats UMAP throughout

    for model in RESNET50_MODELS + VITB16_MODELS:
        if model.startswith("clip") or model == "timm_vit_base_patch16_224.mae":
            continue
        BEST_PARAMS["KMeans"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
        )

    BEST_PARAMS["KMeans"]["clip_RN50"].update(
        {
            "dim_reducer": "PCA",
            "ndim_reduced": 500,
            "zscore": True,
            "pca_variance": None,
        }
    )
    BEST_PARAMS["KMeans"]["clip_vitb16"].update(
        {
            "dim_reducer": "PCA",
            "ndim_reduced": 500,
            "zscore": True,
            "pca_variance": None,
        }
    )
    BEST_PARAMS["KMeans"]["timm_vit_base_patch16_224.mae"].update(
        {
            "dim_reducer": "PCA",
            "pca_variance": 0.85,
            "zscore": True,
            "ndim_reduced": None,
        }
    )

    # AffinityPropagation
    # Use PCA with 10 dims for every encoder except
    # - resnet50 (supervised) : original embeddings, no reduction (AMI=0.62);
    #   perf gets worse if they are whitened (AMI=0.55) and although the perf increases
    #   as num dims are reduced it doesn't quite recover. PCA perf peaks at 10-20 dim (AMI=0.57).
    # - dino_resnet50 : does marginally better at UMAP 50 (AMI=0.52495) than PCA 10 (AMI=0.5044)
    # - timm_vit_base_patch16_224.mae : PCA 0.95 variance explained (AMI=0.303).
    #   Definite improvement from 10 to 20 dims, but not much improvement above that.

    for model in models:
        if model in ["resnet50", "dino_resnet50", "timm_vit_base_patch16_224.mae"]:
            continue
        BEST_PARAMS["AffinityPropagation"][model].update(
            {
                "dim_reducer": "PCA",
                "ndim_reduced": 10,
                "zscore": True,
                "pca_variance": None,
                "dim_reducer_man": "None",
            }
        )

    BEST_PARAMS["AffinityPropagation"]["resnet50"].update(
        {"dim_reducer": "None", "dim_reducer_man": "None", "zscore": False}
    )
    BEST_PARAMS["AffinityPropagation"]["dino_resnet50"].update(
        {
            "dim_reducer": "PCA",
            "pca_variance": 0.95,
            "zscore": True,
            "ndim_reduced": None,
            "dim_reducer_man": "None",
        }
    )
    BEST_PARAMS["AffinityPropagation"]["timm_vit_base_patch16_224.mae"].update(
        {
            "dim_reducer": "PCA",
            "pca_variance": 0.95,
            "zscore": True,
            "ndim_reduced": None,
            "dim_reducer_man": "None",
        }
    )

    # AgglomerativeClustering
    # Use UMAP (num dims unimportant; we select 50d for consistency) for every encoder except
    # - timm_vit_base_patch16_224.mae : PCA 0.98 variance explained (i.e. nearly all
    #   dimensions kept), which is not noticably better than using 500 dim PCA but there is
    #   an increase compared to using less than 500d.

    for model in models:
        if model == "timm_vit_base_patch16_224.mae":
            continue
        BEST_PARAMS["AgglomerativeClustering"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS["AgglomerativeClustering"]["timm_vit_base_patch16_224.mae"].update(
        {
            "dim_reducer": "PCA",
            "pca_variance": 0.98,
            "zscore": True,
            "ndim_reduced": None,
            "dim_reducer_man": "None",
        }
    )

    # HDBSCAN
    # Use UMAP for every encoder except
    # - timm_vit_base_patch16_224.mae : PCA 0.95 variance explained (AMI=0.085) which is
    #   not noticably better than PCA with 50 dim

    for model in models:
        if model in ["timm_vit_base_patch16_224.mae"]:
            continue
        BEST_PARAMS["HDBSCAN"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS["HDBSCAN"]["timm_vit_base_patch16_224.mae"].update(
        {
            "dim_reducer": "PCA",
            "pca_variance": 0.95,
            "zscore": True,
            "ndim_reduced": None,
            "dim_reducer_man": "None",
        }
    )

    # OPTICS
    # Use UMAP for every encoder, no exceptions necessary
    for model in models:
        BEST_PARAMS["OPTICS"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    return BEST_PARAMS


def get_best_params(BEST_PARAMS):
    BEST_PARAMS_v1 = copy.deepcopy(BEST_PARAMS)
    BEST_PARAMS_v1["_version"] = "v1.0"

    BEST_PARAMS_v2 = copy.deepcopy(BEST_PARAMS)
    BEST_PARAMS_v2["_version"] = "v2.0"

    print("Updating dim choices for new method")
    # Updated dim choices
    # (changed to this when we swapped to using weighted average instead of straight
    # average between Imagenet-1k, Imagenette, Imagewoof)

    # Changed KMeans clip_RN50 from PCA 500 to UMAP 50, so it uses fewer dimensions
    # (probably more stable than using 500-d which is what PCA needs to marginally beat UMAP)
    BEST_PARAMS_v2["KMeans"]["clip_RN50"].update(
        {
            "dim_reducer": None,
            "ndim_reduced": None,
            "zscore": False,
            "pca_variance": None,
        }
    )
    BEST_PARAMS_v2["KMeans"]["clip_RN50"].update(
        {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )
    # Changed KMeans MAE from PCA 85% to PCA 200
    # (since we see perf above plateaus at 200-d, there is no point going above that)
    BEST_PARAMS_v2["KMeans"]["timm_vit_base_patch16_224.mae"].update(
        {
            "dim_reducer": "PCA",
            "zscore": True,
            "ndim_reduced": 200,
            "pca_variance": None,
        }
    )
    # Changed KMeans clip_vitb16 from PCA 500 to PCA 75%
    # (gives a notably better train set AMI measurement above)
    BEST_PARAMS_v2["KMeans"]["clip_vitb16"].update(
        {
            "dim_reducer": "PCA",
            "zscore": True,
            "pca_variance": 0.75,
            "ndim_reduced": None,
        }
    )

    # Changed AffinityPropagation dino_resnet50 from PCA 95% to PCA 10
    # (performance is basically equal, so no point using higher-dim space;
    # could have done UMAP 50 instead with basically equal train AMI to PCA 10,
    # but didn't for consistency with other models)
    BEST_PARAMS_v2["AffinityPropagation"]["dino_resnet50"].update(
        {"dim_reducer": "PCA", "zscore": True, "ndim_reduced": 10, "pca_variance": None}
    )
    # Changed AffinityPropagation MAE from PCA 95% to PCA 100
    BEST_PARAMS_v2["AffinityPropagation"]["timm_vit_base_patch16_224.mae"].update(
        {
            "dim_reducer": "PCA",
            "zscore": True,
            "ndim_reduced": 100,
            "pca_variance": None,
        }
    )

    print(
        "Updating dim choices to use Affinity Prop dim results found with 0.9 damping,"
        " prefering PCA reduction by percentage variance explained"
    )
    BEST_PARAMS_v3 = {
        clusterer: {
            model: copy.deepcopy(DEFAULT_PARAMS[clusterer]) for model in ALL_MODELS
        }
        for clusterer in ALL_CLUSTERERS
    }
    BEST_PARAMS_v3["_version"] = "v3.0"

    # KMeans
    for model in RESNET50_MODELS + VITB16_MODELS + FT_MODELS:
        if (
            model == "none"
            or model.startswith("random")
            or model.startswith("clip")
            or model == "timm_vit_base_patch16_224.mae"
        ):
            continue
        BEST_PARAMS_v3["KMeans"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
        )

    BEST_PARAMS_v3["KMeans"]["none"].update(
        {"image_size": 32, "dim_reducer": "PCA", "pca_variance": 0.98, "zscore": True}
    )
    BEST_PARAMS_v3["KMeans"]["random_resnet50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.95, "zscore": True}
    )
    BEST_PARAMS_v3["KMeans"]["random_vitb16"].update(
        {"dim_reducer": "PCA", "ndim_reduced": 100, "zscore": True}
    )

    BEST_PARAMS_v3["KMeans"]["clip_RN50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.85, "zscore": True}
    )
    BEST_PARAMS_v3["KMeans"]["clip_vitb16"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.75, "zscore": True}
    )
    BEST_PARAMS_v3["KMeans"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.85, "zscore": True}
    )

    # AffinityPropagation
    for model in ALL_MODELS:
        BEST_PARAMS_v3["AffinityPropagation"][model].update({"affinity_damping": 0.9})

    for model in [
        "resnet50",
        "clip_RN50",
        "vitb16",
        "mocov3_vit_base",
        "mae_pretrain_vit_base_global",
        "dino_vitb16",
        "clip_vitb16",
    ] + FT_MODELS:
        BEST_PARAMS_v3["AffinityPropagation"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
        )
    for model in ["mocov3_resnet50", "vicreg_resnet50", "dino_resnet50"]:
        BEST_PARAMS_v3["AffinityPropagation"][model].update(
            {
                "dim_reducer_man": "PaCMAP",
                "ndim_reduced_man": 50,
                "dim_reducer_man_nn": None,
            }
        )

    BEST_PARAMS_v3["AffinityPropagation"]["none"].update(
        {"image_size": 32, "dim_reducer": "PCA", "pca_variance": 0.8, "zscore": True}
    )
    BEST_PARAMS_v3["AffinityPropagation"]["random_resnet50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.99, "zscore": True}
    )
    BEST_PARAMS_v3["AffinityPropagation"]["random_vitb16"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.98, "zscore": True}
    )

    BEST_PARAMS_v3["KMeans"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.99, "zscore": True}
    )

    # AgglomerativeClustering
    for model in ALL_MODELS:
        if (
            model == "none"
            or model.startswith("random")
            or model == "timm_vit_base_patch16_224.mae"
        ):
            continue
        BEST_PARAMS_v3["AgglomerativeClustering"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS_v3["AgglomerativeClustering"]["none"].update(
        {"image_size": 32, "dim_reducer": "PCA", "pca_variance": 0.75, "zscore": True}
    )
    BEST_PARAMS_v3["AgglomerativeClustering"]["random_resnet50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.98, "zscore": True}
    )
    BEST_PARAMS_v3["AgglomerativeClustering"]["random_vitb16"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.85, "zscore": True}
    )
    BEST_PARAMS_v3["AgglomerativeClustering"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.98, "zscore": True}
    )

    # HDBSCAN
    for model in ALL_MODELS:
        if model in ["timm_vit_base_patch16_224.mae"]:
            continue
        BEST_PARAMS_v3["HDBSCAN"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS_v3["HDBSCAN"]["none"].update({"image_size": 32})
    BEST_PARAMS_v3["HDBSCAN"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.95, "zscore": True}
    )

    # OPTICS - TODO
    # Use UMAP for every encoder, no exceptions necessary (not checked raw or random)
    for model in ALL_MODELS:
        BEST_PARAMS_v3["OPTICS"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    print(
        "Updating dim choices to use Affinity Prop dim results found with 0.9 damping,"
        " stop PCA at 95%"
    )
    BEST_PARAMS_v4 = {
        clusterer: {
            model: copy.deepcopy(DEFAULT_PARAMS[clusterer]) for model in ALL_MODELS
        }
        for clusterer in ALL_CLUSTERERS
    }
    BEST_PARAMS_v4["_version"] = "v4.0"
    for clusterer in BEST_PARAMS_v4:
        if clusterer.startswith("_"):
            continue
        BEST_PARAMS_v4[clusterer]["none"].update({"image_size": 32})

    # KMeans
    for model in RESNET50_MODELS + VITB16_MODELS + FT_MODELS:
        if (
            model == "none"
            or model.startswith("random")
            or model.startswith("clip")
            or model == "timm_vit_base_patch16_224.mae"
            or model == "mae_pretrain_vit_base_global"
        ):
            continue
        BEST_PARAMS_v4["KMeans"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
        )

    BEST_PARAMS_v4["KMeans"]["none"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.90, "zscore": True}
    )
    BEST_PARAMS_v4["KMeans"]["random_resnet50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.95, "zscore": True}
    )
    BEST_PARAMS_v4["KMeans"]["random_vitb16"].update(
        {"dim_reducer": "PCA", "ndim_reduced": 100, "zscore": True}
    )

    BEST_PARAMS_v4["KMeans"]["clip_RN50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.85, "zscore": True}
    )
    BEST_PARAMS_v4["KMeans"]["clip_vitb16"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.75, "zscore": True}
    )
    BEST_PARAMS_v4["KMeans"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.95, "zscore": True}
    )
    BEST_PARAMS_v4["KMeans"]["mae_pretrain_vit_base_global"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.9, "zscore": True}
    )

    # AffinityPropagation
    for model in ALL_MODELS:
        BEST_PARAMS_v4["AffinityPropagation"][model].update({"affinity_damping": 0.9})

    for model in [
        "resnet50",
        "clip_RN50",
        "vitb16",
        "mocov3_vit_base",
        "mae_pretrain_vit_base_global",
        "dino_vitb16",
        "clip_vitb16",
    ] + FT_MODELS:
        BEST_PARAMS_v4["AffinityPropagation"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
        )
    for model in ["mocov3_resnet50", "vicreg_resnet50", "dino_resnet50"]:
        # tbc
        BEST_PARAMS_v4["AffinityPropagation"][model].update(
            {
                "dim_reducer_man": "UMAP",
                "ndim_reduced_man": 50,
                "dim_reducer_man_nn": None,
            }
        )

    BEST_PARAMS_v4["AffinityPropagation"]["none"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.8, "zscore": True}
    )
    BEST_PARAMS_v4["AffinityPropagation"]["random_resnet50"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.9, "zscore": True}
    )
    BEST_PARAMS_v4["AffinityPropagation"]["random_vitb16"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.9, "zscore": True}
    )
    BEST_PARAMS_v4["AffinityPropagation"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "ndim_reduced": 200, "zscore": True}
    )

    # AgglomerativeClustering
    for model in ALL_MODELS:
        if (
            model == "none"
            or model.startswith("random")
            or model == "timm_vit_base_patch16_224.mae"
            or model == "mae_pretrain_vit_base_global"
        ):
            continue
        BEST_PARAMS_v4["AgglomerativeClustering"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS_v4["AgglomerativeClustering"]["none"].update(
        {"dim_reducer": "PCA", "ndim_reduced": 200, "zscore": True}
    )
    BEST_PARAMS_v4["AgglomerativeClustering"]["random_resnet50"].update(
        {"dim_reducer": "PCA", "ndim_reduced": 200, "zscore": True}
    )
    BEST_PARAMS_v4["AgglomerativeClustering"]["random_vitb16"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.85, "zscore": True}
    )
    BEST_PARAMS_v4["AgglomerativeClustering"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.90, "zscore": True}
    )
    BEST_PARAMS_v4["AgglomerativeClustering"]["mae_pretrain_vit_base_global"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.85, "zscore": True}
    )

    # HDBSCAN
    for model in ALL_MODELS:
        if model in ["timm_vit_base_patch16_224.mae"]:
            continue
        BEST_PARAMS_v4["HDBSCAN"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS_v4["HDBSCAN"]["timm_vit_base_patch16_224.mae"].update(
        {"dim_reducer": "PCA", "pca_variance": 0.95, "zscore": True}
    )

    # OPTICS - TODO
    # Use UMAP for every encoder, no exceptions necessary (not checked raw or random)
    for model in ALL_MODELS:
        BEST_PARAMS_v4["OPTICS"][model].update(
            {"dim_reducer_man": "UMAP", "ndim_reduced_man": 50, "dim_reducer": "None"}
        )

    BEST_PARAMS_v5 = copy.deepcopy(BEST_PARAMS_v4)
    BEST_PARAMS_v5["_version"] = "v5.0"

    BEST_PARAMS_v5["SpectralClustering"]["none"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["random_resnet50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "ndim_reduced": 200,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["resnet50"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["mocov3_resnet50"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["dino_resnet50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.8,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["vicreg_resnet50"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["clip_RN50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.9,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["random_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.95,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.7,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["mocov3_vit_base"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.85,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["dino_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.9,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["timm_vit_base_patch16_224.mae"].update(
        {"zscore": True, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["mae_pretrain_vit_base_global"].update(
        {"zscore": True, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["clip_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.7,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["ft_mocov3_resnet50"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["ft_dino_resnet50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.8,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["ft_vicreg_resnet50"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["SpectralClustering"]["ft_mocov3_vit_base"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.95,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["ft_dino_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.9,
        }
    )
    BEST_PARAMS_v5["SpectralClustering"]["mae_finetuned_vit_base_global"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.75,
        }
    )

    BEST_PARAMS_v5["LouvainCommunities"]["none"].update(
        {"zscore": False, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["random_resnet50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.7,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["resnet50"].update(
        {"dim_reducer": "None", "dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mocov3_resnet50"].update(
        {"dim_reducer": "None", "dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["dino_resnet50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.75,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["vicreg_resnet50"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.9,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["random_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.75,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "ndim_reduced": 10,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mocov3_vit_base"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.75,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["dino_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.75,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["timm_vit_base_patch16_224.mae"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "pca_variance": 0.75,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mae_pretrain_vit_base_global"].update(
        {"zscore": True, "dim_reducer": "None", "dim_reducer_man": "None"}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_mocov3_resnet50"].update(
        {"dim_reducer": "None", "dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_dino_resnet50"].update(
        {"dim_reducer": "None", "dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_vicreg_resnet50"].update(
        {"dim_reducer": "None", "dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )  # adjusted 20 -> 50
    BEST_PARAMS_v5["LouvainCommunities"]["ft_mocov3_vit_base"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "ndim_reduced": 100,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_dino_vitb16"].update(
        {
            "zscore": True,
            "dim_reducer": "PCA",
            "dim_reducer_man": "None",
            "ndim_reduced": 10,
        }
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mae_finetuned_vit_base_global"].update(
        {"dim_reducer": "None", "dim_reducer_man": "UMAP", "ndim_reduced_man": 50}
    )  # adjusted 10 -> 50

    for model in [
        "resnet50",
        "mocov3_resnet50",
        "vicreg_resnet50",
        "vitb16",
        "timm_vit_base_patch16_224.mae",
    ]:
        BEST_PARAMS_v1["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "ward",
            }
        )
    for model in ["dino_resnet50", "clip_RN50", "dino_vitb16"]:
        BEST_PARAMS_v1["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "average",
            }
        )
    for model in ["mocov3_vit_base", "clip_vitb16"]:
        BEST_PARAMS_v1["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "chebyshev",
                "aggclust_linkage": "average",
            }
        )

    # vicreg_resnet50 is the only change from v1 to v2
    for model in [
        "resnet50",
        "mocov3_resnet50",
        "vitb16",
        "timm_vit_base_patch16_224.mae",
    ]:
        BEST_PARAMS_v2["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "ward",
            }
        )
    for model in ["vicreg_resnet50", "dino_resnet50", "clip_RN50", "dino_vitb16"]:
        BEST_PARAMS_v2["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "average",
            }
        )
    for model in ["mocov3_vit_base", "clip_vitb16"]:
        BEST_PARAMS_v2["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "chebyshev",
                "aggclust_linkage": "average",
            }
        )

    for model in ["none", "resnet50", "mocov3_resnet50", "vitb16"] + FT_MODELS:
        BEST_PARAMS_v3["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "ward",
            }
        )
    for model in ["vicreg_resnet50", "dino_resnet50", "clip_RN50", "dino_vitb16"]:
        BEST_PARAMS_v3["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "average",
            }
        )
    for model in ["mocov3_vit_base", "clip_vitb16", "random_resnet50", "random_vitb16"]:
        BEST_PARAMS_v3["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "chebyshev",
                "aggclust_linkage": "average",
            }
        )
    for model in ["timm_vit_base_patch16_224.mae"]:
        BEST_PARAMS_v3["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "cosine",
                "aggclust_linkage": "average",
            }
        )

    # TODO:
    # - mae_pretrain_vit_base_global
    # - clip_vitb16 (leaving as-is for now)
    for model in ALL_MODELS:
        BEST_PARAMS_v4["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "tbd",
                "aggclust_linkage": "tbd",
            }
        )

    for model in ["resnet50", "mocov3_resnet50", "vitb16"] + FT_MODELS:
        BEST_PARAMS_v4["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "ward",
            }
        )
    for model in ["vicreg_resnet50", "dino_resnet50", "clip_RN50", "dino_vitb16"]:
        BEST_PARAMS_v4["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "average",
            }
        )
    for model in ["mocov3_vit_base", "clip_vitb16", "random_resnet50", "random_vitb16"]:
        BEST_PARAMS_v4["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "chebyshev",
                "aggclust_linkage": "average",
            }
        )
    for model in [
        "none",
        "timm_vit_base_patch16_224.mae",
        "mae_pretrain_vit_base_global",
    ]:
        BEST_PARAMS_v4["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "cosine",
                "aggclust_linkage": "average",
            }
        )

    # TODO:
    # - mae_pretrain_vit_base_global
    # - clip_vitb16 (leaving as-is for now)
    for model in ALL_MODELS:
        BEST_PARAMS_v5["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "tbd",
                "aggclust_linkage": "tbd",
            }
        )

    for model in ["resnet50", "mocov3_resnet50", "vitb16"] + FT_MODELS:
        BEST_PARAMS_v5["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "ward",
            }
        )
    for model in ["vicreg_resnet50", "dino_resnet50", "clip_RN50", "dino_vitb16"]:
        BEST_PARAMS_v5["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "euclidean",
                "aggclust_linkage": "average",
            }
        )
    for model in ["mocov3_vit_base", "clip_vitb16", "random_resnet50", "random_vitb16"]:
        BEST_PARAMS_v5["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "chebyshev",
                "aggclust_linkage": "average",
            }
        )
    for model in [
        "none",
        "timm_vit_base_patch16_224.mae",
        "mae_pretrain_vit_base_global",
    ]:
        BEST_PARAMS_v5["AgglomerativeClustering"][model].update(
            {
                "distance_metric": "cosine",
                "aggclust_linkage": "average",
            }
        )

    BEST_PARAMS_v1["AC w/ C"] = copy.deepcopy(BEST_PARAMS_v1["AgglomerativeClustering"])
    BEST_PARAMS_v1["AC w/o C"] = copy.deepcopy(
        BEST_PARAMS_v1["AgglomerativeClustering"]
    )
    BEST_PARAMS_v2["AC w/ C"] = copy.deepcopy(BEST_PARAMS_v2["AgglomerativeClustering"])
    BEST_PARAMS_v2["AC w/o C"] = copy.deepcopy(
        BEST_PARAMS_v2["AgglomerativeClustering"]
    )
    BEST_PARAMS_v3["AC w/ C"] = copy.deepcopy(BEST_PARAMS_v3["AgglomerativeClustering"])
    BEST_PARAMS_v3["AC w/o C"] = copy.deepcopy(
        BEST_PARAMS_v3["AgglomerativeClustering"]
    )
    BEST_PARAMS_v4["AC w/ C"] = copy.deepcopy(BEST_PARAMS_v4["AgglomerativeClustering"])
    BEST_PARAMS_v4["AC w/o C"] = copy.deepcopy(
        BEST_PARAMS_v4["AgglomerativeClustering"]
    )
    BEST_PARAMS_v5["AC w/ C"] = copy.deepcopy(BEST_PARAMS_v5["AgglomerativeClustering"])
    BEST_PARAMS_v5["AC w/o C"] = copy.deepcopy(
        BEST_PARAMS_v5["AgglomerativeClustering"]
    )

    for model in BEST_PARAMS_v1["AC w/ C"]:
        BEST_PARAMS_v1["AC w/ C"][model].update({"aggclust_dist_thresh": None})
    for model in BEST_PARAMS_v2["AC w/ C"]:
        BEST_PARAMS_v2["AC w/ C"][model].update({"aggclust_dist_thresh": None})
    for model in BEST_PARAMS_v3["AC w/ C"]:
        BEST_PARAMS_v3["AC w/ C"][model].update({"aggclust_dist_thresh": None})
    for model in BEST_PARAMS_v4["AC w/ C"]:
        BEST_PARAMS_v4["AC w/ C"][model].update({"aggclust_dist_thresh": None})
    for model in BEST_PARAMS_v5["AC w/ C"]:
        BEST_PARAMS_v5["AC w/ C"][model].update({"aggclust_dist_thresh": None})

    for model in BEST_PARAMS_v2["AC w/o C"]:
        BEST_PARAMS_v2["AC w/o C"][model].update(
            {"zscore2": "average", "ndim_correction": True}
        )
    for model in BEST_PARAMS_v3["AC w/o C"]:
        BEST_PARAMS_v3["AC w/o C"][model].update(
            {"zscore2": "average", "ndim_correction": True}
        )
    for model in BEST_PARAMS_v4["AC w/o C"]:
        BEST_PARAMS_v4["AC w/o C"][model].update(
            {"zscore2": "average", "ndim_correction": True}
        )
    for model in BEST_PARAMS_v5["AC w/o C"]:
        BEST_PARAMS_v5["AC w/o C"][model].update(
            {"zscore2": "average", "ndim_correction": True}
        )

    # Run AgglomerativeClustering experiments with number of clusters unknown
    # 	resnet50        	20.0
    # 	mocov3_resnet50 	20.0
    # 	vicreg_resnet50 	20.0
    # 	vitb16 	            20.0
    # 	dino_resnet50     	 1.0
    # 	clip_RN50 	         1.0
    # 	dino_vitb16 	     2.0
    # 	mocov3_vit_base 	 1.0
    # 	clip_vitb16 	     0.5
    # 	timm_vit_base_patch16_224.mae 	200.0

    for model in ["resnet50", "mocov3_resnet50", "vicreg_resnet50", "vitb16"]:
        BEST_PARAMS_v1["AC w/o C"][model].update({"aggclust_dist_thresh": 20.0})
    for model in ["dino_resnet50", "clip_RN50", "mocov3_vit_base"]:
        BEST_PARAMS_v1["AC w/o C"][model].update({"aggclust_dist_thresh": 1.0})
    BEST_PARAMS_v1["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v1["AC w/o C"]["clip_vitb16"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v1["AC w/o C"]["timm_vit_base_patch16_224.mae"][
        "aggclust_dist_thresh"
    ] = 200.0

    BEST_PARAMS_v2["AC w/o C"]["resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v2["AC w/o C"]["mocov3_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v2["AC w/o C"]["vicreg_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v2["AC w/o C"]["dino_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v2["AC w/o C"]["clip_RN50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v2["AC w/o C"]["vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v2["AC w/o C"]["mocov3_vit_base"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v2["AC w/o C"]["timm_vit_base_patch16_224.mae"][
        "aggclust_dist_thresh"
    ] = 5.0
    BEST_PARAMS_v2["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 0.2
    BEST_PARAMS_v2["AC w/o C"]["clip_vitb16"]["aggclust_dist_thresh"] = 1.0

    BEST_PARAMS_v3["AC w/o C"]["none"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v3["AC w/o C"]["random_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v3["AC w/o C"]["resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v3["AC w/o C"]["mocov3_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v3["AC w/o C"]["dino_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v3["AC w/o C"]["vicreg_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v3["AC w/o C"]["clip_RN50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v3["AC w/o C"]["random_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v3["AC w/o C"]["vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v3["AC w/o C"]["mocov3_vit_base"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v3["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 0.2
    BEST_PARAMS_v3["AC w/o C"]["timm_vit_base_patch16_224.mae"][
        "aggclust_dist_thresh"
    ] = 0.5
    BEST_PARAMS_v3["AC w/o C"]["clip_vitb16"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v3["AC w/o C"]["ft_mocov3_resnet50"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v3["AC w/o C"]["ft_dino_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v3["AC w/o C"]["ft_vicreg_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v3["AC w/o C"]["ft_mocov3_vit_base"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v3["AC w/o C"]["ft_dino_vitb16"]["aggclust_dist_thresh"] = 2.0

    # TODO:
    # - none
    # - random_resnet50
    # - timm_vit_base_patch16_224.mae
    # - mae_pretrain_vit_base_global
    # - clip_vitb16 (leave as-is)
    # - ft_mocov3_resnet50 (tbc)
    # - mae_finetuned_vit_base_global
    BEST_PARAMS_v4["AC w/o C"]["resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mocov3_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v4["AC w/o C"]["dino_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["vicreg_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["random_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mocov3_vit_base"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v4["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 0.2
    BEST_PARAMS_v4["AC w/o C"]["ft_mocov3_resnet50"][
        "aggclust_dist_thresh"
    ] = 2.0  # tbc
    BEST_PARAMS_v4["AC w/o C"]["ft_dino_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_vicreg_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_mocov3_vit_base"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_dino_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["_version"] = "v4.0"

    # TODO:
    # - none
    # - timm_vit_base_patch16_224.mae (tbc)
    # - mae_pretrain_vit_base_global
    # - clip_vitb16 (leave as-is)
    BEST_PARAMS_v4["AC w/o C"]["random_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v4["AC w/o C"]["resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mocov3_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v4["AC w/o C"]["dino_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["vicreg_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["clip_RN50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["random_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mocov3_vit_base"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v4["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 0.2
    BEST_PARAMS_v4["AC w/o C"]["timm_vit_base_patch16_224.mae"][
        "aggclust_dist_thresh"
    ] = 0.5  # tbc
    BEST_PARAMS_v4["AC w/o C"]["clip_vitb16"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v4["AC w/o C"]["ft_mocov3_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_dino_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_vicreg_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_mocov3_vit_base"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_dino_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mae_finetuned_vit_base_global"][
        "aggclust_dist_thresh"
    ] = 2.0
    BEST_PARAMS_v4["_version"] = "v4.1"

    # v4.4
    BEST_PARAMS_v4["AC w/o C"]["none"]["aggclust_dist_thresh"] = 0.71
    BEST_PARAMS_v4["AC w/o C"]["random_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v4["AC w/o C"]["resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mocov3_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v4["AC w/o C"]["dino_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["vicreg_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["clip_RN50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v4["AC w/o C"]["random_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mocov3_vit_base"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v4["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 0.2
    BEST_PARAMS_v4["AC w/o C"]["timm_vit_base_patch16_224.mae"][
        "aggclust_dist_thresh"
    ] = 0.71
    BEST_PARAMS_v4["AC w/o C"]["mae_pretrain_vit_base_global"][
        "aggclust_dist_thresh"
    ] = 0.71
    BEST_PARAMS_v4["AC w/o C"]["clip_vitb16"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v4["AC w/o C"]["ft_mocov3_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_dino_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_vicreg_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_mocov3_vit_base"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["ft_dino_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v4["AC w/o C"]["mae_finetuned_vit_base_global"][
        "aggclust_dist_thresh"
    ] = 2.0
    BEST_PARAMS_v4["_version"] = "v4.4"

    # v5.0
    BEST_PARAMS_v5["AC w/o C"]["none"]["aggclust_dist_thresh"] = 0.71
    BEST_PARAMS_v5["AC w/o C"]["random_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v5["AC w/o C"]["resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["mocov3_resnet50"]["aggclust_dist_thresh"] = 10.0
    BEST_PARAMS_v5["AC w/o C"]["dino_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v5["AC w/o C"]["vicreg_resnet50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v5["AC w/o C"]["clip_RN50"]["aggclust_dist_thresh"] = 0.5
    BEST_PARAMS_v5["AC w/o C"]["random_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["mocov3_vit_base"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v5["AC w/o C"]["dino_vitb16"]["aggclust_dist_thresh"] = 0.2
    BEST_PARAMS_v5["AC w/o C"]["timm_vit_base_patch16_224.mae"][
        "aggclust_dist_thresh"
    ] = 0.71
    BEST_PARAMS_v5["AC w/o C"]["mae_pretrain_vit_base_global"][
        "aggclust_dist_thresh"
    ] = 0.71
    BEST_PARAMS_v5["AC w/o C"]["clip_vitb16"]["aggclust_dist_thresh"] = 1.0
    BEST_PARAMS_v5["AC w/o C"]["ft_mocov3_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["ft_dino_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["ft_vicreg_resnet50"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["ft_mocov3_vit_base"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["ft_dino_vitb16"]["aggclust_dist_thresh"] = 2.0
    BEST_PARAMS_v5["AC w/o C"]["mae_finetuned_vit_base_global"][
        "aggclust_dist_thresh"
    ] = 2.0

    for model in BEST_PARAMS_v1["AffinityPropagation"]:
        BEST_PARAMS_v1["AffinityPropagation"][model]["affinity_damping"] = 0.5
    for model in BEST_PARAMS_v2["AffinityPropagation"]:
        BEST_PARAMS_v2["AffinityPropagation"][model]["affinity_damping"] = 0.5
    for model in BEST_PARAMS_v3["AffinityPropagation"]:
        BEST_PARAMS_v3["AffinityPropagation"][model]["affinity_damping"] = 0.9
    for model in BEST_PARAMS_v4["AffinityPropagation"]:
        BEST_PARAMS_v4["AffinityPropagation"][model]["affinity_damping"] = 0.9

    BEST_PARAMS_v3["AffinityPropagation"]["none"]["affinity_damping"] = 0.85
    BEST_PARAMS_v3["AffinityPropagation"]["random_resnet50"]["affinity_damping"] = 0.5
    BEST_PARAMS_v3["AffinityPropagation"]["resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v3["AffinityPropagation"]["mocov3_resnet50"]["affinity_damping"] = 0.8
    BEST_PARAMS_v3["AffinityPropagation"]["dino_resnet50"]["affinity_damping"] = 0.8
    BEST_PARAMS_v3["AffinityPropagation"]["vicreg_resnet50"]["affinity_damping"] = 0.75
    BEST_PARAMS_v3["AffinityPropagation"]["clip_RN50"]["affinity_damping"] = 0.85
    BEST_PARAMS_v3["AffinityPropagation"]["random_vitb16"]["affinity_damping"] = 0.7
    BEST_PARAMS_v3["AffinityPropagation"]["vitb16"]["affinity_damping"] = 0.9
    BEST_PARAMS_v3["AffinityPropagation"]["mocov3_vit_base"]["affinity_damping"] = 0.75
    BEST_PARAMS_v3["AffinityPropagation"]["dino_vitb16"]["affinity_damping"] = 0.85
    BEST_PARAMS_v3["AffinityPropagation"]["timm_vit_base_patch16_224.mae"][
        "affinity_damping"
    ] = 0.5
    BEST_PARAMS_v3["AffinityPropagation"]["mae_pretrain_vit_base_global"][
        "affinity_damping"
    ] = 0.9  # To match
    BEST_PARAMS_v3["AffinityPropagation"]["clip_vitb16"]["affinity_damping"] = 0.95
    BEST_PARAMS_v3["AffinityPropagation"]["ft_mocov3_resnet50"][
        "affinity_damping"
    ] = 0.9  # Match supervised/ft resnet50
    BEST_PARAMS_v3["AffinityPropagation"]["ft_vicreg_resnet50"][
        "affinity_damping"
    ] = 0.9
    BEST_PARAMS_v3["AffinityPropagation"]["ft_dino_vitb16"]["affinity_damping"] = 0.9
    BEST_PARAMS_v3["AffinityPropagation"]["ft_mocov3_vit_base"][
        "affinity_damping"
    ] = 0.9  # Match supervised/ft resnet50
    BEST_PARAMS_v3["AffinityPropagation"]["mae_finetuned_vit_base_global"][
        "affinity_damping"
    ] = 0.9

    BEST_PARAMS_v4["AffinityPropagation"]["none"]["affinity_damping"] = 0.85
    BEST_PARAMS_v4["AffinityPropagation"]["random_resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["mocov3_resnet50"]["affinity_damping"] = 0.75
    BEST_PARAMS_v4["AffinityPropagation"]["dino_resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["vicreg_resnet50"]["affinity_damping"] = 0.8
    BEST_PARAMS_v4["AffinityPropagation"]["clip_RN50"]["affinity_damping"] = 0.85
    BEST_PARAMS_v4["AffinityPropagation"]["random_vitb16"]["affinity_damping"] = 0.95
    BEST_PARAMS_v4["AffinityPropagation"]["vitb16"]["affinity_damping"] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["mocov3_vit_base"]["affinity_damping"] = 0.75
    BEST_PARAMS_v4["AffinityPropagation"]["dino_vitb16"]["affinity_damping"] = 0.85
    BEST_PARAMS_v4["AffinityPropagation"]["timm_vit_base_patch16_224.mae"][
        "affinity_damping"
    ] = 0.6
    BEST_PARAMS_v4["AffinityPropagation"]["mae_pretrain_vit_base_global"][
        "affinity_damping"
    ] = 0.6
    BEST_PARAMS_v4["AffinityPropagation"]["clip_vitb16"]["affinity_damping"] = 0.95
    BEST_PARAMS_v4["AffinityPropagation"]["ft_mocov3_resnet50"][
        "affinity_damping"
    ] = 0.95
    BEST_PARAMS_v4["AffinityPropagation"]["ft_dino_resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["ft_vicreg_resnet50"][
        "affinity_damping"
    ] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["ft_mocov3_vit_base"][
        "affinity_damping"
    ] = 0.95
    BEST_PARAMS_v4["AffinityPropagation"]["ft_dino_vitb16"]["affinity_damping"] = 0.9
    BEST_PARAMS_v4["AffinityPropagation"]["mae_finetuned_vit_base_global"][
        "affinity_damping"
    ] = 0.9

    BEST_PARAMS_v5["AffinityPropagation"]["none"]["affinity_damping"] = 0.85
    BEST_PARAMS_v5["AffinityPropagation"]["random_resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["mocov3_resnet50"]["affinity_damping"] = 0.75
    BEST_PARAMS_v5["AffinityPropagation"]["dino_resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["vicreg_resnet50"]["affinity_damping"] = 0.8
    BEST_PARAMS_v5["AffinityPropagation"]["clip_RN50"]["affinity_damping"] = 0.85
    BEST_PARAMS_v5["AffinityPropagation"]["random_vitb16"]["affinity_damping"] = 0.95
    BEST_PARAMS_v5["AffinityPropagation"]["vitb16"]["affinity_damping"] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["mocov3_vit_base"]["affinity_damping"] = 0.75
    BEST_PARAMS_v5["AffinityPropagation"]["dino_vitb16"]["affinity_damping"] = 0.85
    BEST_PARAMS_v5["AffinityPropagation"]["timm_vit_base_patch16_224.mae"][
        "affinity_damping"
    ] = 0.6
    BEST_PARAMS_v5["AffinityPropagation"]["mae_pretrain_vit_base_global"][
        "affinity_damping"
    ] = 0.6
    BEST_PARAMS_v5["AffinityPropagation"]["clip_vitb16"]["affinity_damping"] = 0.95
    BEST_PARAMS_v5["AffinityPropagation"]["ft_mocov3_resnet50"][
        "affinity_damping"
    ] = 0.95
    BEST_PARAMS_v5["AffinityPropagation"]["ft_dino_resnet50"]["affinity_damping"] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["ft_vicreg_resnet50"][
        "affinity_damping"
    ] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["ft_mocov3_vit_base"][
        "affinity_damping"
    ] = 0.95
    BEST_PARAMS_v5["AffinityPropagation"]["ft_dino_vitb16"]["affinity_damping"] = 0.9
    BEST_PARAMS_v5["AffinityPropagation"]["mae_finetuned_vit_base_global"][
        "affinity_damping"
    ] = 0.9

    for model in RESNET50_MODELS + VITB16_MODELS:
        BEST_PARAMS_v1["HDBSCAN"][model].update(
            {
                "distance_metric": "euclidean",
                "hdbscan_method": "eom",
            }
        )

    for model in RESNET50_MODELS + VITB16_MODELS:
        BEST_PARAMS_v2["HDBSCAN"][model].update(
            {
                "distance_metric": "euclidean",
                "hdbscan_method": "eom",
            }
        )
    for model in [
        "vicreg_resnet50",
        "dino_resnet50",
        "clip_RN50",
        "dino_vitb16",
        "clip_vitb16",
    ]:
        BEST_PARAMS_v2["HDBSCAN"][model].update(
            {
                "distance_metric": "l1",
            }
        )
    BEST_PARAMS_v2["HDBSCAN"]["vitb16"]["distance_metric"] = "chebyshev"

    for model in [
        "resnet50",
        "mocov3_resnet50",
        "mocov3_vit_base",
        "timm_vit_base_patch16_224.mae",
    ]:
        BEST_PARAMS_v3["HDBSCAN"][model].update(
            {
                "distance_metric": "euclidean",
                "hdbscan_method": "eom",
            }
        )

    for model in [
        "random_resnet50",
        "vicreg_resnet50",
        "dino_resnet50",
        "clip_RN50",
        "random_vitb16",
        "dino_vitb16",
        "clip_vitb16",
    ]:
        BEST_PARAMS_v3["HDBSCAN"][model].update(
            {
                "distance_metric": "l1",
                "hdbscan_method": "eom",
            }
        )

    for model in ["vitb16"]:
        BEST_PARAMS_v3["HDBSCAN"][model].update(
            {
                "distance_metric": "chebyshev",
                "hdbscan_method": "eom",
            }
        )

    BEST_PARAMS_v4["HDBSCAN"]["none"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v4["HDBSCAN"]["none"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["random_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["random_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["resnet50"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v4["HDBSCAN"]["resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["mocov3_resnet50"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v4["HDBSCAN"]["mocov3_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["dino_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["dino_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["vicreg_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["vicreg_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["clip_RN50"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["clip_RN50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["random_vitb16"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["random_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["vitb16"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v4["HDBSCAN"]["vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["mocov3_vit_base"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v4["HDBSCAN"]["mocov3_vit_base"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["dino_vitb16"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["dino_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["timm_vit_base_patch16_224.mae"][
        "distance_metric"
    ] = "euclidean"
    BEST_PARAMS_v4["HDBSCAN"]["timm_vit_base_patch16_224.mae"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["mae_pretrain_vit_base_global"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["mae_pretrain_vit_base_global"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["clip_vitb16"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["clip_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["ft_mocov3_resnet50"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v4["HDBSCAN"]["ft_mocov3_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["ft_dino_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v4["HDBSCAN"]["ft_dino_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["ft_vicreg_resnet50"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v4["HDBSCAN"]["ft_vicreg_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["ft_mocov3_vit_base"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v4["HDBSCAN"]["ft_mocov3_vit_base"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["ft_dino_vitb16"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v4["HDBSCAN"]["ft_dino_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v4["HDBSCAN"]["mae_finetuned_vit_base_global"][
        "distance_metric"
    ] = "chebyshev"
    BEST_PARAMS_v4["HDBSCAN"]["mae_finetuned_vit_base_global"]["hdbscan_method"] = "eom"

    BEST_PARAMS_v5["HDBSCAN"]["none"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v5["HDBSCAN"]["none"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["random_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["random_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["resnet50"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v5["HDBSCAN"]["resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["mocov3_resnet50"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v5["HDBSCAN"]["mocov3_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["dino_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["dino_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["vicreg_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["vicreg_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["clip_RN50"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["clip_RN50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["random_vitb16"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["random_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["vitb16"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v5["HDBSCAN"]["vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["mocov3_vit_base"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v5["HDBSCAN"]["mocov3_vit_base"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["dino_vitb16"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["dino_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["timm_vit_base_patch16_224.mae"][
        "distance_metric"
    ] = "euclidean"
    BEST_PARAMS_v5["HDBSCAN"]["timm_vit_base_patch16_224.mae"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["mae_pretrain_vit_base_global"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["mae_pretrain_vit_base_global"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["clip_vitb16"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["clip_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["ft_mocov3_resnet50"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v5["HDBSCAN"]["ft_mocov3_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["ft_dino_resnet50"]["distance_metric"] = "l1"
    BEST_PARAMS_v5["HDBSCAN"]["ft_dino_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["ft_vicreg_resnet50"]["distance_metric"] = "euclidean"
    BEST_PARAMS_v5["HDBSCAN"]["ft_vicreg_resnet50"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["ft_mocov3_vit_base"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v5["HDBSCAN"]["ft_mocov3_vit_base"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["ft_dino_vitb16"]["distance_metric"] = "chebyshev"
    BEST_PARAMS_v5["HDBSCAN"]["ft_dino_vitb16"]["hdbscan_method"] = "eom"
    BEST_PARAMS_v5["HDBSCAN"]["mae_finetuned_vit_base_global"][
        "distance_metric"
    ] = "chebyshev"
    BEST_PARAMS_v5["HDBSCAN"]["mae_finetuned_vit_base_global"]["hdbscan_method"] = "eom"

    BEST_PARAMS_v5["SpectralClustering"]["none"]["spectral_n_neighbors"] = 10
    BEST_PARAMS_v5["SpectralClustering"]["random_resnet50"]["spectral_n_neighbors"] = 50
    BEST_PARAMS_v5["SpectralClustering"]["resnet50"]["spectral_n_neighbors"] = 20
    BEST_PARAMS_v5["SpectralClustering"]["mocov3_resnet50"]["spectral_n_neighbors"] = 30
    BEST_PARAMS_v5["SpectralClustering"]["dino_resnet50"]["spectral_n_neighbors"] = 10
    BEST_PARAMS_v5["SpectralClustering"]["vicreg_resnet50"]["spectral_n_neighbors"] = 10
    BEST_PARAMS_v5["SpectralClustering"]["clip_RN50"]["spectral_n_neighbors"] = 30
    BEST_PARAMS_v5["SpectralClustering"]["random_vitb16"]["spectral_n_neighbors"] = 50
    BEST_PARAMS_v5["SpectralClustering"]["vitb16"]["spectral_n_neighbors"] = 30
    BEST_PARAMS_v5["SpectralClustering"]["mocov3_vit_base"]["spectral_n_neighbors"] = 50
    BEST_PARAMS_v5["SpectralClustering"]["dino_vitb16"]["spectral_n_neighbors"] = 10
    BEST_PARAMS_v5["SpectralClustering"]["timm_vit_base_patch16_224.mae"][
        "spectral_n_neighbors"
    ] = 10
    BEST_PARAMS_v5["SpectralClustering"]["mae_pretrain_vit_base_global"][
        "spectral_n_neighbors"
    ] = 30
    BEST_PARAMS_v5["SpectralClustering"]["clip_vitb16"]["spectral_n_neighbors"] = 20
    BEST_PARAMS_v5["SpectralClustering"]["ft_mocov3_resnet50"][
        "spectral_n_neighbors"
    ] = 30
    BEST_PARAMS_v5["SpectralClustering"]["ft_dino_resnet50"][
        "spectral_n_neighbors"
    ] = 20
    BEST_PARAMS_v5["SpectralClustering"]["ft_vicreg_resnet50"][
        "spectral_n_neighbors"
    ] = 20
    BEST_PARAMS_v5["SpectralClustering"]["ft_mocov3_vit_base"][
        "spectral_n_neighbors"
    ] = 50
    BEST_PARAMS_v5["SpectralClustering"]["ft_dino_vitb16"]["spectral_n_neighbors"] = 50
    BEST_PARAMS_v5["SpectralClustering"]["mae_finetuned_vit_base_global"][
        "spectral_n_neighbors"
    ] = 50

    BEST_PARAMS_v5["LouvainCommunities"]["none"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["random_resnet50"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["resnet50"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mocov3_resnet50"].update(
        {"distance_metric": "l1", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["dino_resnet50"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["vicreg_resnet50"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["random_vitb16"].update(
        {"distance_metric": "l1", "louvain_remove_self_loops": True}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["vitb16"].update(
        {"distance_metric": "chebyshev", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mocov3_vit_base"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["dino_vitb16"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["timm_vit_base_patch16_224.mae"].update(
        {"distance_metric": "l1", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mae_pretrain_vit_base_global"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_mocov3_resnet50"].update(
        {"distance_metric": "l1", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_dino_resnet50"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_vicreg_resnet50"].update(
        {"distance_metric": "chebyshev", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_mocov3_vit_base"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": True}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_dino_vitb16"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mae_finetuned_vit_base_global"].update(
        {"distance_metric": "l2", "louvain_remove_self_loops": False}
    )

    BEST_PARAMS_v5["LouvainCommunities"]["none"].update({"louvain_resolution": 1.2})
    BEST_PARAMS_v5["LouvainCommunities"]["random_resnet50"].update(
        {"louvain_resolution": 2.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["resnet50"].update(
        {"louvain_resolution": 1.0}
    )  # Tied - used default
    BEST_PARAMS_v5["LouvainCommunities"]["mocov3_resnet50"].update(
        {"louvain_resolution": 1.4}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["dino_resnet50"].update(
        {"louvain_resolution": 1.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["vicreg_resnet50"].update(
        {"louvain_resolution": 1.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["random_vitb16"].update(
        {"louvain_resolution": 3.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["vitb16"].update({"louvain_resolution": 1.1})
    BEST_PARAMS_v5["LouvainCommunities"]["mocov3_vit_base"].update(
        {"louvain_resolution": 1.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["dino_vitb16"].update(
        {"louvain_resolution": 1.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["timm_vit_base_patch16_224.mae"].update(
        {"louvain_resolution": 1.1}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mae_pretrain_vit_base_global"].update(
        {"louvain_resolution": 1.1}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_mocov3_resnet50"].update(
        {"louvain_resolution": 1.0}
    )  # Tied - used default
    BEST_PARAMS_v5["LouvainCommunities"]["ft_dino_resnet50"].update(
        {"louvain_resolution": 1.0}
    )  # Tied - used default
    BEST_PARAMS_v5["LouvainCommunities"]["ft_vicreg_resnet50"].update(
        {"louvain_resolution": 1.0}
    )  # Tied - used default
    BEST_PARAMS_v5["LouvainCommunities"]["ft_mocov3_vit_base"].update(
        {"louvain_resolution": 1.0}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["ft_dino_vitb16"].update(
        {"louvain_resolution": 1.2}
    )
    BEST_PARAMS_v5["LouvainCommunities"]["mae_finetuned_vit_base_global"].update(
        {"louvain_resolution": 1.0}
    )  # Tied - used default

    BEST_PARAMS = BEST_PARAMS_v5

    return BEST_PARAMS
