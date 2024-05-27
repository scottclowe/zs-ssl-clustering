import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics
import torchvision

from param_setup import (
    CLUSTERER2SH,
    DEFAULT_PARAMS,
    MODEL_GROUPS,
    get_best_params,
    setup_best_params,
)
from utils import (
    fetch_results,
    find_differing_columns,
    fixup_filter,
    get_pred_path,
    select_rows,
)

annotation_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "full"]

attrs_d = {ann_level: [] for ann_level in annotation_levels}

if not os.path.isfile("./inat_labels.npz"):
    inat_ds = torchvision.datasets.INaturalist(
        os.path.expanduser("~/Datasets"),
        version="2021_valid",
        target_type="full",
        download=False,
    )

    print(attrs_d)
    for i in range(len(inat_ds.index)):
        cat_id, fname = inat_ds.index[i]
        for annotation_level in annotation_levels:
            if annotation_level == "full":
                attrs_d[annotation_level].append(cat_id)
            else:
                attrs_d[annotation_level].append(
                    inat_ds.categories_map[cat_id][annotation_level]
                )

    attrs = np.stack(
        [attrs_d[ann_level] for ann_level in annotation_levels],
        axis=-1,
    )

    np.savez("inat_labels.npz", attrs)
else:
    attrs = np.load("./inat_labels.npz", allow_pickle=True)["arr_0"]
    print("LOADED")

for i_attr in range(len(annotation_levels)):
    print(annotation_levels[i_attr], len(np.unique(attrs[:, i_attr])))

print(attrs[np.random.choice(10000, 10),], attrs.shape)


if not os.path.isfile(os.path.join(".", "test_runs.csv")):
    test_runs_df, config_keys = fetch_results()
    test_runs_df.to_csv("test_runs.csv", sep=";")
    with open("config_keys.txt", "wb") as f:
        pickle.dump(config_keys, f)
else:
    test_runs_df = pd.read_csv(os.path.join(".", "test_runs.csv"), sep=";")
    with open("config_keys.txt", "rb") as f:
        my_set = pickle.load(f)

metric_key = "AMI"  # AMI  num_cluster_pred  silhouette-euclidean_pred  silhouette-og-euclidean_pred
show_pc = True
show_fmt = "{:4.0f}"
highlight_best = True
use_si_num = False
eps = 0.005
override_fields = {
    "predictions_dir": "y_pred",
}

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

dataset = "inaturalist"

TEST_ATTRS = annotation_levels
print(TEST_ATTRS)

res_dict = {"backbone": [], "cluster": [], "level": [], "AMI": []}

for backbone in ["ViT-B", "ResNet-50"]:
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
            if "clip" in model:
                continue
            filter = {"model": model, "dataset": dataset}
            sdf = select_rows(test_runs_df, filter, allow_missing=False)
            filter2 = dict(DEFAULT_PARAMS["all"], **BEST_PARAMS[clusterer][model])
            filter2 = {k: v for k, v in filter2.items() if k not in filter}
            filter2.update(filter)
            filter2 = fixup_filter(filter2)
            filter2.update(my_override_fields)

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

            # print(model, dataset, get_pred_path(sdf.iloc[0]))
            try:
                y_pred = np.load("../" + get_pred_path(sdf.iloc[0]))["y_pred"]
            except FileNotFoundError:
                continue
            for i_attr, attr in enumerate(TEST_ATTRS):
                if metric_key.lower() != "ami":
                    raise NotImplementedError()
                my_val = sklearn.metrics.adjusted_mutual_info_score(
                    attrs[:, i_attr], y_pred
                )
                print(model, dataset, clusterer, attr[i_attr], my_val)
                if np.isnan(my_val):
                    continue
                res_dict["backbone"].append(model)
                res_dict["cluster"].append(clusterer)
                res_dict["level"].append(attr[i_attr])
                res_dict["AMI"].append(my_val)

                pred_df = pd.DataFrame.from_dict(res_dict)
                pred_df.to_csv(
                    os.path.join(
                        ".",
                        "inat_analysis.csv",
                    ),
                    index=False,
                )


print()
print("Done!")
print()
print(pred_df)
print()
print()
