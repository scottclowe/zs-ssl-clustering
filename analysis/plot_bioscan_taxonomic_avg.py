import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reformat_backbone(backbone):
    if (
        "resnet50" in backbone
        or "resnet" in backbone
        or "RN" in backbone
        or "ResNet" in backbone
    ):
        d_backbone = "ResNet"
    elif (
        "vit" in backbone
        or "vitb" in backbone
        or "vit-b" in backbone
        or "ViT" in backbone
    ):
        d_backbone = "ViT"
    else:
        print(backbone)

    return d_backbone


def reformat_encoder(model):
    if "vicreg" in model.lower():
        mode = "vicreg"
    elif "mae" in model.lower():
        mode = "mae"
        if "global" in model.lower():
            mode += "-global"
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

    return mode


mode_color_dict = {
    "mae": "tab:blue",
    "mae-global": "tab:brown",
    "supervised": "tab:red",
    "mocov3": "tab:green",
    "vicreg": "tab:orange",
    "dino": "tab:purple",
    "random": "tab:pink",
}

backbone_marker_dict = {"ViT": "^", "ResNet": "s", "none": "o"}

label_dict = {
    "supervised": "X-Ent.",
    "mocov3": "MoCo-v3",
    "dino": "DINO",
    "mae": "MAE (CLS)",
    "mae-global": "MAE (avg)",
    "vicreg": "VICReg",
    "random": "Random",
}


org_df = pd.read_csv("bioscan_analysis.csv")

org_df = org_df.assign(
    encoder_mode=[reformat_encoder(x) for x in org_df["backbone"].values]
)
org_df = org_df.assign(
    backbone=[reformat_backbone(x) for x in org_df["backbone"].values]
)
org_df = org_df.assign(ids=org_df["backbone"] + "-" + org_df["encoder_mode"])

x_labels = ["order", "family", "subfamily", "tribe", "genus", "species", "uri"]

print(org_df)

df_dict = {
    "encoder_mode": [],
    "backbone": [],
    "level": [],
    "AMI": [],
    "AMI_std": [],
    "ids": [],
}
for ids in org_df["ids"].unique():
    df_ids = org_df[org_df["ids"] == ids]

    if df_ids["encoder_mode"].values[0] == "random":
        continue

    for level in x_labels:
        df_level = df_ids[df_ids["level"] == level]

        if len(df_level) != 5:
            print(ids, len(df_level))
            print(df_level["cluster"])
        ami_level = df_level["AMI"].values

        df_dict["encoder_mode"].append(df_level["encoder_mode"].values[0])
        df_dict["backbone"].append(df_level["backbone"].values[0])
        df_dict["level"].append(level)
        df_dict["ids"].append(df_level["ids"].values[0])
        df_dict["AMI"].append(np.mean(ami_level))
        df_dict["AMI_std"].append(np.std(ami_level))


df = pd.DataFrame.from_dict(df_dict)


x_labels = ["order", "family", "subfamily", "tribe", "genus", "species", "BIN"]

print(df)

encoders = df["encoder_mode"].unique()

print(encoders)

fig, axs = plt.subplots(1, 1, sharey=True, figsize=(6, 3))

x_ranges = np.arange(7)


for ids in df["ids"].unique():
    subdf = df[df["ids"] == ids]

    backbone = subdf["backbone"].values[0]
    encoder = subdf["encoder_mode"].values[0]
    ami = subdf["AMI"].values * 100

    axs.plot(
        x_ranges,
        ami,
        color=mode_color_dict[encoder],
        marker=backbone_marker_dict[backbone],
        zorder=6,
    )
    axs.scatter(
        x_ranges,
        ami,
        color=mode_color_dict[encoder],
        marker=backbone_marker_dict[backbone],
        zorder=5,
    )


axs.set_ylim(0, 40)
axs.grid(True, axis="y", which="major", linestyle="-", alpha=0.8)
axs.grid(True, axis="y", which="minor", linestyle="--", alpha=0.3)
axs.set_xticks(x_ranges)
axs.set_xticklabels(x_labels, rotation=45)
axs.set_ylabel("AMI (%)")


encoders_sorted = ["supervised", "mocov3", "dino", "vicreg", "mae", "mae-global"]

labels = [label_dict[x] for x in encoders_sorted]


def colorMarker(m):
    return plt.plot([], [], color=m)[0]


handles = [colorMarker(mode_color_dict[name]) for name in encoders_sorted]


def backboneMarker(m):
    return plt.plot([], [], marker=m, ls="none", color="black")[0]


handles.extend(
    [backboneMarker(backbone_marker_dict[name]) for name in ["ResNet", "ViT"]]
)

labels += ["RN50", "ViT-B"]

ncols = 3
axs.legend(
    handles,
    labels,
    bbox_to_anchor=(0.0, 1.1, 1.0, 0.402),
    loc="lower left",
    ncol=ncols,
    mode="expand",
    borderaxespad=0.0,
)

fig.savefig("bioscan_taxonomic_performance_avg.pdf", bbox_inches="tight")
