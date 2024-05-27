import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

finetuned = "N"
df = pd.read_csv(f"knn_ami_results_{finetuned}.csv")

k_values = df["k"].unique()
clusterers = df["clusterer"].unique()

print(k_values)
print(clusterers)

clusterers_label = [x.replace("AffinityPropagation", "AP") for x in clusterers]

spearman_cor_mat_vit = np.zeros((len(k_values), len(clusterers)))
spearman_cor_mat_resnet = np.zeros((len(k_values), len(clusterers)))
spearman_cor_mat = np.zeros((len(k_values), len(clusterers)))

for k_idx, k in enumerate(k_values):
    for cluster_idx, cluster in enumerate(clusterers):
        subdf = df[(df["k"] == k) & (df["clusterer"] == cluster)]
        acc_values = subdf["ACC"] / 100
        ami_values = subdf["AMI"]

        spearman_cor_mat[k_idx, cluster_idx] = scipy.stats.spearmanr(
            acc_values, ami_values
        ).statistic

        subdf_resnet = subdf[subdf["backbone"] == "ResNet"]
        acc_values = subdf_resnet["ACC"] / 100
        ami_values = subdf_resnet["AMI"]

        spearman_cor_mat_resnet[k_idx, cluster_idx] = scipy.stats.spearmanr(
            acc_values, ami_values
        ).statistic

        subdf_vit = subdf[subdf["backbone"] == "ViT"]
        acc_values = subdf_vit["ACC"] / 100
        ami_values = subdf_vit["AMI"]

        spearman_cor_mat_vit[k_idx, cluster_idx] = scipy.stats.spearmanr(
            acc_values, ami_values
        ).statistic

spearman_cor_mat = spearman_cor_mat.T
spearman_cor_mat_resnet = spearman_cor_mat_resnet.T
spearman_cor_mat_vit = spearman_cor_mat_vit.T

print(spearman_cor_mat_resnet)


print(
    " \\\\\n".join(
        [" & ".join(map(str, line)) for line in spearman_cor_mat_resnet.round(2)]
    )
)

print(
    " \\\\\n".join(
        [" & ".join(map(str, line)) for line in spearman_cor_mat_vit.round(2)]
    )
)
