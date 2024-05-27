import argparse
import os
import pickle

import numpy as np
import pandas as pd
import umap

from param_setup import (
    DEFAULT_PARAMS,
    MODEL_GROUPS,
    TEST_DATASETS,
    get_best_params,
    setup_best_params,
)
from utils import fetch_results, fixup_filter, select_rows


def sanitize_fields(model):
    if "resnet50" in model or "resnet" in model or "RN" in model or "ResNet" in model:
        d_backbone = "ResNet"
    elif "vit" in model or "vitb" in model or "vit-b" in model or "ViT" in model:
        d_backbone = "ViT"
    elif "none" in model:
        d_backbone = "none"
    else:
        print(model)
        return None

    if "vicreg" in model.lower():
        mode = "vicreg"
    elif "mae" in model.lower():
        mode = "mae"
        if "global" in model.lower():
            mode += "-global"
        if "cls" in model.lower():
            mode += "-cls"
    elif "moco" in model.lower():
        mode = "mocov3"
    elif "dino" in model.lower():
        mode = "dino"
    elif "dinov2" in model.lower():
        mode = "dinov2"
        print(model)
    elif "clip" in model.lower():
        mode = "clip"
    elif "random" in model.lower():
        mode = "random"
    else:
        mode = "supervised"

    return d_backbone, mode


def main(args):
    if not os.path.isfile(args.test_run_path):
        test_runs_df, config_keys = fetch_results()
        test_runs_df.to_csv("test_runs.csv", sep=";")
        with open("config_keys.txt", "wb") as f:
            pickle.dump(config_keys, f)
    else:
        test_runs_df = pd.read_csv(args.test_run_path, sep=";")
        with open("config_keys.txt", "rb") as f:
            config_keys = pickle.load(f)
            print(config_keys)

    test_runs_df = test_runs_df[test_runs_df["predictions_dir"] == "y_pred"]

    override_fields = {}

    backbones = ["ResNet-50", "ViT-B", "ResNet-50 [FT]", "ViT-B [FT]"]
    BEST_PARAMS = get_best_params(setup_best_params())
    CLUSTERERS = [
        "KMeans",
        "AC w/ C",
        "AC w/o C",
        "AffinityPropagation",
        "HDBSCAN",
        "SpectralClustering",
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    res_dict = {
        "dataset": [],
        "backbone": [],
        "encoder-mode": [],
        "finetuned": [],
        "clusterer": [],
        "AMI": [],
        "OG-S": [],
        "UMAP-S": [],
        "WANDB": [],
        "id": [],
        "duplicate": [],
    }

    for backbone in backbones:
        for clusterer in CLUSTERERS:
            for dataset in TEST_DATASETS:
                for model in list(MODEL_GROUPS[backbone]):
                    if "clip" in model.lower():
                        continue

                    filter1 = {"model": model, "dataset": dataset}
                    filter2 = dict(
                        DEFAULT_PARAMS["all"], **BEST_PARAMS[clusterer][model]
                    )
                    filter2.update(filter1)
                    filter2.update(override_fields)
                    filter2 = fixup_filter(filter2)

                    if filter2["dim_reducer_man"] != "UMAP":
                        continue

                    sdf = select_rows(test_runs_df, filter2, allow_missing=False)

                    if (
                        len(set(sdf["AMI"].values)) == 1
                        and len(set(sdf["silhouette-og-euclidean_pred"].values)) == 1
                        and len(set(sdf["silhouette-euclidean_pred"].values)) == 1
                    ):
                        for i_sdf in range(len(sdf)):
                            d_backbone, mode = sanitize_fields(model)
                            res_dict["dataset"].append(dataset)
                            res_dict["backbone"].append(d_backbone)
                            res_dict["encoder-mode"].append(mode)
                            res_dict["clusterer"].append(clusterer)
                            res_dict["AMI"].append(sdf["AMI"].values[i_sdf])
                            res_dict["OG-S"].append(
                                sdf["silhouette-og-euclidean_pred"].values[i_sdf]
                            )
                            res_dict["UMAP-S"].append(
                                sdf["silhouette-euclidean_pred"].values[i_sdf]
                            )

                            if "ft" in model.lower() or "finetuned" in model.lower():
                                res_dict["finetuned"].append("Y")
                            else:
                                res_dict["finetuned"].append("N")

                            res_dict["WANDB"].append("Y")
                            res_dict["id"].append(sdf["id"].values[i_sdf])
                            if i_sdf > 0:
                                res_dict["duplicate"].append("Y")
                            else:
                                res_dict["duplicate"].append("N")
                    else:
                        print(f"{model}, {dataset}, {clusterer} -- {len(sdf)}\n\t{sdf}")
                        print(
                            f"{sdf['AMI'].values}-- {sdf['silhouette-og-euclidean_pred'].values} -- {sdf['silhouette-euclidean_pred'].values}"
                        )

    wandb_pred_df = pd.DataFrame.from_dict(res_dict)

    for f in os.listdir(args.silhouette_path):
        method_id = int(f.split("__")[-1][:-4])
        row = test_runs_df[test_runs_df["id"] == method_id]
        wandb_row = wandb_pred_df[wandb_pred_df["id"] == method_id]

        model = row["model"].values[0]
        dataset = row["dataset_name"].values[0]
        clusterer = row["clusterer_name"].values[0]

        if clusterer == "AgglomerativeClustering":
            if np.isnan(row["aggclust_dist_thresh"].values[0]):
                clusterer = "AC w/ C"
            else:
                clusterer = "AC w/o C"

        filter1 = {"model": model, "dataset": dataset}
        filter2 = dict(DEFAULT_PARAMS["all"], **BEST_PARAMS[clusterer][model])
        filter2.update(filter1)
        filter2.update(override_fields)
        filter2 = fixup_filter(filter2)

        sdf = select_rows(row, filter2, allow_missing=False)

        if len(sdf) == 0:
            print(
                f"MISSING - {model} -- {dataset} -- {clusterer} -- {row['aggclust_dist_thresh'].values}"
            )
            continue

        with open(os.path.join(args.silhouette_path, f), "r") as f:
            umap_silhouette = float(f.read())

        if len(wandb_row) > 0:
            continue
            # print(f"{model} -- {dataset} -- {method_id}\n\t{wandb_row}")
            # print(f"\t{umap_silhouette} -- {wandb_row['UMAP-S'].values}")

        d_backbone, mode = sanitize_fields(model)
        res_dict["dataset"].append(dataset)
        res_dict["backbone"].append(d_backbone)
        res_dict["encoder-mode"].append(mode)
        res_dict["clusterer"].append(clusterer)
        res_dict["AMI"].append(row["AMI"].values[0])
        res_dict["OG-S"].append(row["silhouette-og-euclidean_pred"].values[0])
        res_dict["UMAP-S"].append(umap_silhouette)

        if "ft" in model.lower() or "finetuned" in model.lower():
            res_dict["finetuned"].append("Y")
        else:
            res_dict["finetuned"].append("N")

        res_dict["WANDB"].append("N")
        res_dict["id"].append(method_id)
        if i_sdf > 0:
            res_dict["duplicate"].append("Y")
        else:
            res_dict["duplicate"].append("N")

    pred_df = pd.DataFrame.from_dict(res_dict)
    pred_df = pred_df.sort_values(
        by=["dataset", "backbone", "encoder-mode", "finetuned"]
    )
    pred_df.to_csv(
        os.path.join(
            args.output_dir,
            "silhouette_results.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Silhouette Score extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--test_run_path", default="./test_runs.csv", type=str)
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument(
        "--silhouette_path",
        default="C:/Users/aaulab/Downloads/ss_umap/test__z1.0",
        type=str,
    )
    args = parser.parse_args()

    main(args)
