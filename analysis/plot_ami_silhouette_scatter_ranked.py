import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from param_setup import CLUSTERER2COLORRGB, CLUSTERER2SH

FIGS_DIR = "./figs"
os.makedirs(FIGS_DIR, exist_ok=True)

import scipy.stats

CLUSTERERS = [
    "KMeans",
    "SpectralClustering",
    "AC w/ C",
    "AC w/o C",
    "AffinityPropagation",
    "HDBSCAN",
]

fig, ax = plt.subplots(2, len(CLUSTERERS), sharey=True, figsize=(12, 4))

df = pd.read_csv("./silhouette_results.csv")
print(df)

df = df[df["duplicate"] == "N"]
df = df[df["encoder-mode"] != "random"]
df = df[df["encoder-mode"] != "clip"]
# df = df[df["encoder-mode"] != "supervised"]
df = df[df["dataset"] != "in9mixedsame"]
# df = df[df["finetuned"] == "N"]
print(df)

for clusterer in CLUSTERERS:
    row = df[df["clusterer"] == clusterer]
    datasets = row["dataset"].unique()
    encoders = row["encoder-mode"].unique()

    encoder_count = {
        encoder: len(row[row["encoder-mode"] == encoder]["dataset"])
        for encoder in encoders
    }

    print(f"{clusterer} -- {len(datasets)} -- {len(encoders)}")
    print(f"\t{datasets}")
    print(f"\t{encoders}")
    print(f"\t{encoder_count}")

    if clusterer == "KMeans":
        print(row[row["encoder-mode"] == "mae"])
    print()

print()
print()

ami_scores = {
    clusterer: df[df["clusterer"] == clusterer]["AMI"] for clusterer in CLUSTERERS
}
og_s_scores = {
    clusterer: df[df["clusterer"] == clusterer]["OG-S"] for clusterer in CLUSTERERS
}
umap_s_scores = {
    clusterer: df[df["clusterer"] == clusterer]["UMAP-S"] for clusterer in CLUSTERERS
}

ami_scores_sorted = {
    k: scipy.stats.rankdata(v, nan_policy="omit") - 1 for k, v in ami_scores.items()
}

og_s_scores_sorted = {
    k: scipy.stats.rankdata(v, nan_policy="omit") - 1 for k, v in og_s_scores.items()
}

umap_s_scores_sorted = {
    k: scipy.stats.rankdata(v, nan_policy="omit") - 1 for k, v in umap_s_scores.items()
}

print(ami_scores["KMeans"].values)
print(ami_scores_sorted["KMeans"])
print(ami_scores["KMeans"].values[ami_scores_sorted["KMeans"] == 0])
print(ami_scores["KMeans"].values[ami_scores_sorted["KMeans"] == 444])
# print(ami_scores["SpectralClustering"].values)
# print(ami_scores_sorted["SpectralClustering"])

max_datapoints = np.max([len(v) for v in ami_scores_sorted.values()])


my_cols = np.concatenate(
    [
        np.tile(
            CLUSTERER2COLORRGB.get(clusterer, "k"),
            [len(ami_scores[clusterer]), 1],
        )
        for clusterer in CLUSTERERS
    ]
)

cors = []

for i_metric, (metric, metric_sorted) in enumerate(
    [
        (og_s_scores, og_s_scores_sorted),
        (umap_s_scores, umap_s_scores_sorted),
    ]
):
    for i_clusterer, clusterer in enumerate(CLUSTERERS):
        if i_clusterer == len(CLUSTERERS):
            ax[i_metric, i_clusterer].scatter(
                metric_sorted,
                ami_scores_sorted,
                color=my_cols,
                s=20,
                alpha=0.5,
            )
            sel = (~np.isnan(ami_scores)) & (~np.isnan(metric_sorted))
            cor = scipy.stats.spearmanr(ami_scores[sel], metric_sorted[sel]).statistic
            print(f"{clusterer}: {cor:.4f}")
        else:
            ax[i_metric, i_clusterer].scatter(
                metric_sorted[clusterer],
                ami_scores_sorted[clusterer],
                color=np.tile(
                    CLUSTERER2COLORRGB.get(clusterer, "k"),
                    [len(ami_scores[clusterer]), 1],
                ),
                s=20,
                alpha=0.5,
            )
            sel = (~np.isnan(ami_scores[clusterer])) & (~np.isnan(metric[clusterer]))
            cor = scipy.stats.spearmanr(
                ami_scores[clusterer][sel], metric[clusterer][sel]
            ).statistic
            print(
                f"{clusterer}: {cor:.4f} -- {len(ami_scores[clusterer])} / {max_datapoints}"
            )
            ax[i_metric, i_clusterer].text(
                10, max_datapoints - max_datapoints // 10, rf"$\rho={cor:.2f}$"
            )

        if i_clusterer == 0:
            if i_metric == 0:
                ax[i_metric, 0].set_ylabel("Original Embeddings\nrank(AMI)")
            else:
                ax[i_metric, 0].set_ylabel("UMAP-Reduced\nrank(AMI)")

        if i_metric == 1:
            ax[i_metric, i_clusterer].set_xlabel(r"rank($S$)")

        if i_metric == 0:
            title = clusterer
            title = title.replace("AffinityPropagation", "AP")
            title = title.replace("SpectralClustering", "Spectral")
            ax[i_metric, i_clusterer].set_title(title)
        ax[i_metric, i_clusterer].set_xlim(-5, max_datapoints + 21)
        ax[i_metric, i_clusterer].set_ylim(-5, max_datapoints + 21)
        ax[i_metric, i_clusterer].set_aspect("equal")

    def label_fn(c, marker):
        return plt.plot([], [], color=c, ls="None", marker=marker, linewidth=6)[0]

    handles = [
        label_fn(CLUSTERER2COLORRGB.get(clusterer), "o") for clusterer in CLUSTERERS
    ]
    data_labels = [CLUSTERER2SH.get(c, c) for c in CLUSTERERS]

fig.savefig(
    os.path.join(FIGS_DIR, "scatter__emb_space_silhouette_ranking_new.pdf"),
    bbox_inches="tight",
)
