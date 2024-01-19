import os

import pandas as pd

res_folder = "./knn_probe_output"


results_dict = {
    "dataset": [],
    "backbone": [],
    "encoder-mode": [],
    "k": [],
    "top1": [],
    "top5": [],
}


for f in os.listdir(res_folder):
    df = pd.read_csv(os.path.join(res_folder, f))

    results_dict["dataset"].extend([f.split("_")[0]] * len(df))
    results_dict["top1"].extend(df["top1"].to_list())
    results_dict["top5"].extend(df["top5"].to_list())
    results_dict["k"].extend(df["k"].to_list())

    if "resnet50" in f or "resnet" in f or "RN" in f:
        backbone = ["ResNet"] * len(df)
    elif "vit" in f or "vitb" in f or "vit-b" in f:
        backbone = ["ViT"] * len(df)
    else:
        print(f)

    results_dict["backbone"].extend(backbone)

    if "vicreg" in f.lower():
        mode = "vicreg"
    elif "mae" in f.lower():
        mode = "mae"
    elif "moco" in f.lower():
        mode = "mocov3"
    elif "dino" in f.lower():
        mode = "dino"
    elif "dinov2" in f.lower():
        mode = "dinov2"
        print(f)
    elif "clip" in f.lower():
        mode = "clip"
    elif "random" in f.lower():
        mode = "random"
    else:
        mode = "supervised"

    results_dict["encoder-mode"].extend([mode] * len(df))

print(df)

for k in results_dict.keys():
    print(k, len(results_dict[k]))

res_df = pd.DataFrame.from_dict(results_dict)
print(res_df)

res_df.to_csv(os.path.join(".", "knn_results.csv"), index=False)
