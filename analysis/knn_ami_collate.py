import os

import pandas as pd

ami_path = "C:/Users/aaulab/Documents/ZSSSLClust_Dev/analysis/ami_results.csv"
knn_path = "C:/Users/aaulab/Documents/ZSSSLClust_Dev/zs_ssl_probing/knn_results.csv"


ami_df = pd.read_csv(ami_path)
knn_df = pd.read_csv(knn_path)

ami_df = ami_df.assign(model=ami_df.backbone + "-" + ami_df["encoder-mode"])
knn_df = knn_df.assign(model=knn_df.backbone + "-" + knn_df["encoder-mode"])

print(ami_df)
print(knn_df)
finetuned = "Y"
ami_df = ami_df[ami_df["finetuned"] == finetuned]
knn_df = knn_df[knn_df["finetuned"] == finetuned]


print(ami_df)
print(knn_df)

ami_datasets = set(ami_df["dataset"].unique())
knn_datasets = set(knn_df["dataset"].unique())

dataset_intersection = ami_datasets.intersection(knn_datasets)
ami_dataset_only = ami_datasets.difference(knn_datasets)
knn_dataset_only = knn_datasets.difference(ami_datasets)

print(dataset_intersection)
print(ami_dataset_only)
print(knn_dataset_only)

skip_dataset = ["imagenet-o", "imagenet-r", "imagenet-sketch", "imagenetv2"]
skip_mode = ["clip"]


res_dict = {
    "dataset": [],
    "backbone": [],
    "encoder-mode": [],
    "k": [],
    "clusterer": [],
    "ACC": [],
    "AMI": [],
}

for dataset in list(dataset_intersection):
    if dataset in skip_dataset:
        continue

    print(dataset)
    ami_dataset_df = ami_df[ami_df["dataset"] == dataset]
    knn_dataset_df = knn_df[knn_df["dataset"] == dataset]

    ami_models = set(ami_dataset_df["model"].unique())
    knn_models = set(knn_dataset_df["model"].unique())

    model_intersection = ami_models.intersection(knn_models)
    ami_model_only = ami_models.difference(knn_models)
    knn_model_only = knn_models.difference(ami_models)

    print(f"\t AMI and KNN models - {model_intersection}")
    print(f"\t AMI only models - {ami_model_only}")
    print(f"\t KNN only models - {knn_model_only}")
    for model in model_intersection:
        backbone = model.split("-")[0]
        mode = model.split("-")[1]

        if mode in skip_mode:
            continue

        ami_model_df = ami_dataset_df[ami_dataset_df["model"] == model]
        knn_model_df = knn_dataset_df[knn_dataset_df["model"] == model]

        k_values = knn_model_df["k"].values
        clusterers = ami_model_df["clusterer"].values
        for k in k_values:
            k_acc = knn_model_df[knn_model_df["k"] == k]["top1"].values[0]
            for clusterer in clusterers:
                cluster_ami = ami_model_df[ami_model_df["clusterer"] == clusterer][
                    "AMI"
                ].values[0]

                res_dict["dataset"].append(dataset)
                res_dict["backbone"].append(backbone)
                res_dict["encoder-mode"].append(mode)
                res_dict["k"].append(k)
                res_dict["clusterer"].append(clusterer)
                res_dict["ACC"].append(k_acc)
                res_dict["AMI"].append(cluster_ami)


res_df = pd.DataFrame.from_dict(res_dict)
print(res_df)

res_df.to_csv(os.path.join(".", f"knn_ami_results_{finetuned}.csv"), index=False)
