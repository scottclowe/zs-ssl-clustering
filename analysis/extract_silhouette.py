import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tqdm
import umap
from sklearn.metrics import silhouette_score

from param_setup import DEFAULT_PARAMS, get_best_params, setup_best_params
from utils import fetch_results, fixup_filter, select_rows


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

    res_dict = {
        "dataset": [],
        "backbone": [],
        "encoder": [],
        "encoder-mode": [],
        "finetuned": [],
        "clusterer": [],
        "AMI": [],
        "UMAP-S": [],
    }

    if os.path.isfile(pred_path):
        id = int(os.path.splitext(os.path.basename(pred_path))[0].split("__")[-1])
        rows = test_runs_df[test_runs_df["id"] == id]
        if len(rows) == 0:
            raise ValueError(f"ID {id} not found in test_runs.csv")
        if len(rows) != 1:
            print(rows)
            raise ValueError(f"ID {id} found multiple times in test_runs.csv")
        row = rows.iloc[0]
        feat_paths = [f"{row['dataset_name']}__{row['model']}.npz"]
    elif os.path.isdir(pred_path):
        feat_paths = os.listdir(feat_path)
        if not args.output_csv:
            import datetime

            dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.output_csv = f"silhouette_scores_{dtstr}.csv"
    else:
        raise ValueError(f"Invalid path: {pred_path}")

    for i_file, f in tqdm.tqdm(
        enumerate(feat_paths), total=len(feat_paths), disable=len(feat_paths) == 1
    ):
        if "clip" in f:
            continue

        dataset = f.split("_")[0]
        model = f[len(dataset) + 2 : -4]

        if len(feat_paths) > 1 and dataset in ["imagenette", "imagewoof"]:
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

        if os.path.isfile(pred_path):
            if not args.perfile_output_dir:
                raise ValueError(
                    "perfile_output_dir must be specified if prediction_path is a file"
                )
            pred_file = pred_path
            y_pred = np.load(pred_file)["y_pred"]
            print("Computing silhouette score")
            umap_score = silhouette_score(umap_embeddings, y_pred)
            outfname = os.path.join(
                args.perfile_output_dir,
                os.path.splitext(os.path.basename(pred_file))[0] + ".txt",
            )
            print(f"Writing silhouette score to {outfname}")
            if os.path.dirname(outfname):
                os.makedirs(os.path.dirname(outfname), exist_ok=True)
            with open(outfname, "w") as hf:
                hf.write(f"{umap_score}\n")
            return umap_score

        filter = {
            "model": model,
            "dataset": dataset,
            "predictions_dir": "y_pred",
        }

        for clusterer in CLUSTERERS:
            if model not in BEST_PARAMS[clusterer]:
                print(f"Skipping {model} for {clusterer}")
                continue
            sdf = select_rows(test_runs_df, filter, allow_missing=False)
            filter2 = dict(DEFAULT_PARAMS["all"], **BEST_PARAMS[clusterer][model])
            filter2 = {k: v for k, v in filter2.items() if k not in filter}
            filter2.update(filter)
            filter2 = fixup_filter(filter2)
            sdf = select_rows(sdf, filter2, allow_missing=False)
            ids = sdf["id"].values

            umap_silhouette_scores = []
            for i in ids:
                pred_file = os.path.join(pred_path, f"test-{dataset}__{model}__{i}.npz")
                if not os.path.isfile(pred_file):
                    continue
                y_pred = np.load(pred_file)["y_pred"]
                try:
                    umap_score = silhouette_score(umap_embeddings, y_pred)
                except Exception as err:
                    print(
                        f"Error computing UMAP silhouette score for {pred_file}: {err}"
                    )
                    continue
                if args.perfile_output_dir:
                    outfname = os.path.join(
                        args.perfile_output_dir,
                        os.path.splitext(os.path.basename(pred_file))[0] + ".txt",
                    )
                    if os.path.dirname(outfname):
                        os.makedirs(os.path.dirname(outfname), exist_ok=True)
                    with open(outfname, "w") as hf:
                        hf.write(f"{umap_score}\n")

            ami_score = np.nanmedian(sdf["AMI"])
            umap_silhouette_score = np.nanmedian(umap_silhouette_scores)

            if "resnet" in model.lower() or "RN" in model:
                d_backbone = "ResNet"
            elif "vit" in model.lower():
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
            res_dict["encoder"].append(model)
            res_dict["encoder-mode"].append(mode)
            res_dict["clusterer"].append(clusterer)
            res_dict["AMI"].append(ami_score)
            res_dict["UMAP-S"].append(umap_silhouette_score)

            if "ft" in model.lower() or "finetuned" in model.lower():
                res_dict["finetuned"].append("Y")
            else:
                res_dict["finetuned"].append("N")

        if args.output_csv:  # and (i_file % 50 == 0 or i_file == len(feat_paths) - 1):
            status = "intermediate" if i_file != len(feat_paths) - 1 else "final"
            print(f"Saving {status} results")
            pred_df = pd.DataFrame.from_dict(res_dict)
            if os.path.dirname(args.output_csv):
                os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            pred_df.to_csv(args.output_csv, index=False)
            print(f"Saved {status} results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Silhouette Score extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--test_run_path", default="./test_runs.csv", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument(
        "--feature_path",
        default="embeddings/test__z1.0",
        type=str,
    )
    parser.add_argument(
        "--prediction_path",
        default="y_pred/test__z1.0",
        type=str,
    )
    parser.add_argument(
        "--perfile_output_dir",
        default="ss_umap/test__z1.0",
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
