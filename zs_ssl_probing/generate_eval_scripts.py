import os

local_train_folder = "C:/Users/aaulab/Downloads/zs_ssl_train"
local_test_folder = "C:/Users/aaulab/Downloads/zs_ssl_test"


commands = []

cloud_folder = "/home/create.aau.dk/joha/zsssl_embeddings"

imagenet_alternatives = ["imagenet-o", "imagenet-r", "imagenet-sketch", "imagenetv2"]


for f in os.listdir(local_train_folder):
    command_dummy = f"python eval_knn.py --use_cuda --load_features {f}"
    commands.append(command_dummy)

    if "imagenet__" in f:
        for i_a in imagenet_alternatives:
            alt_test = f.replace("imagenet", i_a)
            command_dummy = f"python eval_knn.py --use_cuda --load_features {f} --load_features_test {alt_test}"
            commands.append(command_dummy)


print(commands)

with open("knn_commands.sh", "w") as f:
    for line in commands:
        f.write(f"{line}\n")
