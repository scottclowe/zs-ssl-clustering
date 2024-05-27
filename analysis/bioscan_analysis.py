import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics

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

column_dtypes = {
    "sampleid": str,
    "processid": str,
    "uri": str,
    "name": "category",
    "phylum": str,
    "class": str,
    "order": str,
    "family": str,
    "subfamily": str,
    "tribe": str,
    "genus": str,
    "species": str,
    "subspecies": str,
    "nucraw": str,
    "image_file": str,
    "large_diptera_family": "category",
    "medium_diptera_family": "category",
    "small_diptera_family": "category",
    "large_insect_order": "category",
    "medium_insect_order": "category",
    "small_insect_order": "category",
    "chunk_number": "uint8",
    "copyright_license": "category",
    "copyright_holder": "category",
    "copyright_institution": "category",
    "copyright_contact": "category",
    "photographer": "category",
    "author": "category",
}

usecols = [
    "sampleid",
    "uri",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "tribe",
    "genus",
    "species",
    "image_file",
    "chunk_number",
]


def load_bioscan_metadata():
    df = pd.read_csv(
        "C:/Users/aaulab/Downloads/BIOSCAN_Insect_Dataset_metadata.tsv",
        sep="\t",
        dtype=column_dtypes,
        usecols=usecols,
    )
    # Convert missing values to NaN
    df.replace("not_classified", pd.NA, inplace=True)
    # df[df == "not_classified"] = pd.NA
    # Fix some tribe labels which were only partially applied
    df.loc[df["genus"].notna() & (df["genus"] == "Asteia"), "tribe"] = "Asteiini"
    df.loc[df["genus"].notna() & (df["genus"] == "Nemorilla"), "tribe"] = "Winthemiini"
    df.loc[df["genus"].notna() & (df["genus"] == "Philaenus"), "tribe"] = "Philaenini"
    # Add missing genus labels
    sel = df["genus"].isna() & df["species"].notna()
    df.loc[sel, "genus"] = df.loc[sel, "species"].apply(lambda x: x.split(" ")[0])
    # Add placeholder for missing tribe labels
    sel = df["tribe"].isna() & df["genus"].notna()
    sel2 = df["subfamily"].notna()
    df.loc[sel & sel2, "tribe"] = "unassigned " + df.loc[sel, "subfamily"]
    df.loc[sel & ~sel2, "tribe"] = "unassigned " + df.loc[sel, "family"]
    # Add placeholder for missing subfamily labels
    sel = df["subfamily"].isna() & df["tribe"].notna()
    df.loc[sel, "subfamily"] = "unassigned " + df.loc[sel, "family"]
    # Convert label columns to category dtype; add index columns to use for targets
    label_cols = [
        "phylum",
        "class",
        "order",
        "family",
        "subfamily",
        "tribe",
        "genus",
        "species",
        "uri",
    ]
    for c in label_cols:
        df[c] = df[c].astype("category")
        df[c + "_index"] = df[c].cat.codes
    print(df)
    return df


def bioscan_partition(metadata, partition_name):
    if partition_name == "train":
        partition_files = ["train_seen", "test_unseen_keys"]
    elif partition_name == "val":
        partition_files = ["test_seen", "seen_keys", "test_unseen"]
    elif partition_name == "test":
        partition_files = [
            "seen_keys",
            "test_seen",
            "test_unseen",
            "test_unseen_keys",
        ]
    else:
        raise ValueError(f"Unrecognized partition name: {partition_name}")

    partition_samples = []
    for fname in partition_files:
        with open(os.path.join(".", "bioscan_partitions", fname + ".txt"), "r") as f:
            partition_samples += f.readlines()

    partition_samples = [x.rstrip() for x in partition_samples]
    metadata = metadata.loc[metadata["sampleid"].isin(partition_samples)]

    return metadata


def main():
    annotation_levels = [
        "order",
        "family",
        "subfamily",
        "tribe",
        "genus",
        "species",
        "uri",
    ]

    if not os.path.isfile("./bioscan_labels.npz"):
        df = bioscan_partition(load_bioscan_metadata(), "test")

        attrs_d = {}
        # Biological Taxonomy
        for taxa in annotation_levels:
            attrs_d[taxa] = df[f"{taxa}_index"].to_list()

        attrs = np.stack(
            [attrs_d[ann_level] for ann_level in list(attrs_d.keys())],
            axis=-1,
        )

        np.savez("bioscan_labels.npz", attrs)
    else:
        attrs = np.load("./bioscan_labels.npz", allow_pickle=True)["arr_0"]
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

    metric_key = "AMI"  # AMI  num_cluster_pred  silhouette-euclidean_pred  silhouette-og-euclidean_pred
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

    dataset = "bioscan1m"

    TEST_ATTRS = annotation_levels
    print(TEST_ATTRS)

    res_dict = {"backbone": [], "cluster": [], "level": [], "AMI": []}

    for backbone in ["ViT-B", "ResNet-50"]:
        print(MODEL_GROUPS[backbone])
        first_agg = True
        for clusterer in CLUSTERERS:
            my_override_fields = override_fields.copy()
            if (
                first_agg
                and clusterer == "AgglomerativeClustering"
                and metric_key != "num_cluster_pred"
            ):
                first_agg = False
                my_override_fields["aggclust_dist_thresh"] = None
            elif clusterer == "AgglomerativeClustering":
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
                    print()
                    print(filter2)
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
                            "bioscan_analysis.csv",
                        ),
                        index=False,
                    )

    print()
    print("Done!")
    print()
    print(pred_df)
    print()
    print()


if __name__ == "__main__":
    main()
