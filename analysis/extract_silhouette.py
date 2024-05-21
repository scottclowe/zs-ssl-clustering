import argparse
import os
import pickle

import numpy as np
import pandas as pd
import umap
from param_setup import (
    DEFAULT_PARAMS,
    fetch_results,
    get_best_params,
    setup_best_params,
)
from sklearn.metrics import silhouette_score

from utils import fixup_filter, select_rows


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

    BEST_PARAMS = get_best_params(setup_best_params())
    CLUSTERERS = [
        "KMeans",
        "AC w/ C",
        "AC w/o C",
        "AffinityPropagation",
        "HDBSCAN",
        "SpectralClustering",
    ]

    feat_path = args.feature_path
    pred_path = args.prediction_path

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
    }

    total_files = len(os.listdir(feat_path))
    for f_idx, f in enumerate(os.listdir(feat_path)):
        if f_idx % 100 == 0:
            print(f"{f_idx} / {total_files}")

        if "clip" in f:
            continue

        dataset = f.split("_")[0]
        model = f[len(dataset) + 2 : -4]

        if dataset in ["imagenette", "imagewoof"]:
            continue

        data = np.load(os.path.join(feat_path, f))
        embeddings = data["embeddings"]

        reducer_man = umap.UMAP(
            n_neighbors=args.dim_reducer_man_nn,
            n_components=args.ndim_reduced_man,
            min_dist=args.min_dist,
            metric=args.distance_metric_man,
            random_state=args.seed,
            n_jobs=args.workers,  # Only 1 worker used if RNG is manually seeded
            verbose=args.verbose > 0,
        )

        umap_embeddings = reducer_man.fit_transform(embeddings)

        filter = {
            "model": model,
            "dataset": dataset,
        }

        for clusterer in CLUSTERERS:
            sdf = select_rows(test_runs_df, filter, allow_missing=False)
            filter2 = dict(DEFAULT_PARAMS["all"], **BEST_PARAMS[clusterer][model])
            filter2 = {k: v for k, v in filter2.items() if k not in filter}
            filter2.update(filter)
            filter2 = fixup_filter(filter2)
            sdf = select_rows(sdf, filter2, allow_missing=False)
            ids = sdf["id"].values

            og_silhouette_scores = []
            umap_silhouette_scores = []
            for i in ids:
                y_pred = np.load(
                    os.path.join(pred_path, f"test-{dataset}__{model}__{i}.npz")
                )["y_pred"]

                og_silhouette_scores.append(silhouette_score(embeddings, y_pred))
                umap_silhouette_scores.append(silhouette_score(umap_embeddings, y_pred))

            ami_score = np.nanmedian(sdf["AMI"])
            og_silhouette_score = np.nanmedian(og_silhouette_scores)
            umap_silhouette_score = np.nanmedian(umap_silhouette_scores)

            if (
                "resnet50" in model
                or "resnet" in model
                or "RN" in model
                or "ResNet" in model
            ):
                d_backbone = "ResNet"
            elif (
                "vit" in model or "vitb" in model or "vit-b" in model or "ViT" in model
            ):
                d_backbone = "ViT"
            elif "none" in model:
                d_backbone = "none"
            else:
                print(model)
                continue

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

            res_dict["dataset"].append(dataset)
            res_dict["backbone"].append(d_backbone)
            res_dict["encoder-mode"].append(mode)
            res_dict["clusterer"].append(clusterer)
            res_dict["AMI"].append(ami_score)
            res_dict["OG-S"].append(og_silhouette_score)
            res_dict["UMAP-S"].append(umap_silhouette_score)

            if "ft" in model.lower() or "finetuned" in model.lower():
                res_dict["finetuned"].append("Y")
            else:
                res_dict["finetuned"].append("N")

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
        "--feature_path",
        default="C:/Users/aaulab/Downloads/ZS_SSL_Embeddings/zs_ssl_test-UMAP",
        type=str,
    )
    parser.add_argument(
        "--prediction_path",
        default="C:/Users/aaulab/Downloads/zsc_neurips/y_pred(2)/y_pred/test__z1.0",
        type=str,
    )

    parser.add_argument("--dim_reducer_man_nn", default=30, type=int)
    parser.add_argument("--ndim_reduced_man", default=50, type=int)
    parser.add_argument("--distance_metric_man", default="l2", type=str)
    parser.add_argument("--min_dist", default=0.0, type=float)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    main(args)
