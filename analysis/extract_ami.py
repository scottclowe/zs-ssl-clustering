import os
import pickle

import numpy as np
import pandas as pd

from param_setup import (
    CLUSTERER2SH,
    DEFAULT_PARAMS,
    MODEL_GROUPS,
    TEST_DATASETS,
    get_best_params,
    setup_best_params,
)
from utils import fetch_results, find_differing_columns, fixup_filter, select_rows

if not os.path.isfile(os.path.join(".", "test_runs.csv")):
    test_runs_df, config_keys = fetch_results()
    test_runs_df.to_csv("test_runs.csv", sep=";")
    with open("config_keys.txt", "wb") as f:
        pickle.dump(config_keys, f)
else:
    test_runs_df = pd.read_csv(os.path.join(".", "test_runs.csv"), sep=";")
    with open("config_keys.txt", "rb") as f:
        config_keys = pickle.load(f)
        print(config_keys)


metric_key = "AMI"
override_fields = {}
BEST_PARAMS = get_best_params(setup_best_params())

CLUSTERERS = [
    "KMeans",
    "AC w/ C",
    "AC w/o C",
    "AffinityPropagation",
    "HDBSCAN",
    "SpectralClustering",
]

print(MODEL_GROUPS)


res_dict = {
    "dataset": [],
    "backbone": [],
    "encoder-mode": [],
    "finetuned": [],
    "clusterer": [],
    "AMI": [],
}

for backbone in ["all"]:
    print(MODEL_GROUPS[backbone])

    first_agg = True
    for clusterer in CLUSTERERS:
        clusterername = CLUSTERER2SH.get(clusterer, clusterer)
        my_override_fields = override_fields.copy()
        if (
            first_agg
            and clusterer == "AgglomerativeClustering"
            and metric_key != "num_cluster_pred"
        ):
            first_agg = False
            my_override_fields["aggclust_dist_thresh"] = None
            clusterername = "AC  w/ C"
        elif clusterer == "AgglomerativeClustering":
            clusterername = "AC w/o C"
            if "aggclust_dist_thresh" in my_override_fields:
                del my_override_fields["aggclust_dist_thresh"]

        for model in list(MODEL_GROUPS[backbone]):
            if "clip" in model.lower():
                continue
            if "random" in model.lower():
                continue
            if "none" in model.lower():
                continue
            for dataset in TEST_DATASETS:
                filter = {
                    "model": model,
                    "dataset": dataset,
                }
                sdf = select_rows(test_runs_df, filter, allow_missing=False)
                filter2 = dict(DEFAULT_PARAMS["all"], **BEST_PARAMS[clusterer][model])
                filter2 = {k: v for k, v in filter2.items() if k not in filter}
                filter2.update(filter)
                filter2.update(my_override_fields)
                filter2 = fixup_filter(filter2)
                sdf = select_rows(sdf, filter2, allow_missing=False)
                if len(sdf) < 1:
                    print(
                        f"No data for {model}-{dataset}-{clusterer}"
                    )  # \n{filter} {filter2}")
                    continue
                if len(sdf) > 1:
                    perf = sdf.iloc[0][metric_key]
                    if sum(sdf[metric_key] != perf) > 0:
                        print()
                        print(
                            f"More than one result with {metric_key} values",
                            list(sdf[metric_key]),
                        )
                        print(f"for search {filter}\nand {filter2}")
                        dif_cols = find_differing_columns(sdf, config_keys)
                        print(f"columns which differ: {dif_cols}")
                        if dif_cols:
                            for col in dif_cols:
                                print(f"  {col}: {list(sdf[col])}")

                print(sdf)
                my_val = np.nanmedian(sdf[metric_key])
                if np.isnan(my_val):
                    continue

                if (
                    "resnet50" in model
                    or "resnet" in model
                    or "RN" in model
                    or "ResNet" in model
                ):
                    d_backbone = "ResNet"
                elif (
                    "vit" in model
                    or "vitb" in model
                    or "vit-b" in model
                    or "ViT" in model
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
                res_dict["AMI"].append(my_val)

                if "ft" in model.lower() or "finetuned" in model.lower():
                    res_dict["finetuned"].append("Y")
                else:
                    res_dict["finetuned"].append("N")

pred_df = pd.DataFrame.from_dict(res_dict)
pred_df = pred_df.sort_values(by=["dataset", "backbone", "encoder-mode", "finetuned"])
pred_df.to_csv(
    os.path.join(
        ".",
        "ami_results.csv",
    ),
    index=False,
)
