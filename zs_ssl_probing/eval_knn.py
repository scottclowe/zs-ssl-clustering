# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import datasets


def relabel(cs_train, cs_test):
    cs_train = cs_train.copy()
    cs_test = cs_test.copy()
    unq_idx = np.unique(np.concatenate([cs_train, cs_test]))
    d = {idx: unq_idx[idx] for idx in range(len(unq_idx))}
    for i in range(len(cs_train)):
        j = cs_train[i]
        cs_train[i] = d[j]
    for i in range(len(cs_test)):
        j = cs_test[i]
        cs_test[i] = d[j]
    print(d)
    return cs_train, cs_test


@torch.no_grad()
def knn_classifier(
    train_features, train_labels, test_features, test_labels, k, T, num_classes=1000
):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    imgs_per_chunk = 500
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    aggregated_predictions = None

    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        if args.use_cuda:
            features = features.cuda()
            targets = targets.cuda()

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        if aggregated_predictions is None:
            aggregated_predictions = predictions[:, 0]
        else:
            aggregated_predictions = torch.concat(
                [aggregated_predictions, predictions[:, 0]], dim=0
            )

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (
            top5 + correct.narrow(1, 0, min(5, k)).sum().item()
        )  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    return top1, top5, aggregated_predictions.cpu().numpy()


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation with weighted k-NN on ImageNet")
    parser.add_argument(
        "--nb_knn",
        default=[1, 10, 20, 100, 200],
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM",
    )
    parser.add_argument(
        "--load_features",
        default=None,
        help="""If the features have
        already been computed, where to find them.""",
    )
    parser.add_argument(
        "--load_features_test",
        default=None,
        help="""If the features have
        already been computed, where to find them.""",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="""If the features have
        already been computed, where to find them.""",
    )
    parser.add_argument(
        "--output_dir_pred",
        default=None,
        help="""If the features have
        already been computed, where to find them.""",
    )
    parser.add_argument(
        "--train_feat_dir",
        default="/home/create.aau.dk/joha/zsssl_embeddings/zs_ssl_train/",
        help="""If the features have
        already been computed, where to find them.""",
    )
    parser.add_argument(
        "--test_feat_dir",
        default="/home/create.aau.dk/joha/zsssl_embeddings/zs_ssl_test/",
        help="""If the features have
        already been computed, where to find them.""",
    )
    args = parser.parse_args()

    print(
        "/n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    if args.output_dir is None:
        args.output_dir = "./knn_probe_output"

    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)

    if args.output_dir_pred is None:
        args.output_dir_pred = "./knn_probe_predictions"

    os.makedirs(args.output_dir_pred, exist_ok=True)
    print(args.output_dir_pred)

    train_load_features = args.load_features
    if args.load_features_test is not None:
        test_load_features = args.load_features_test
    else:
        test_load_features = args.load_features

    train_dict = np.load(
        os.path.join(args.train_feat_dir, train_load_features), allow_pickle=True
    )
    test_dict = np.load(
        os.path.join(args.test_feat_dir, test_load_features), allow_pickle=True
    )

    train_features = torch.tensor(train_dict["embeddings"], dtype=torch.float32)
    train_labels = train_dict["y_true"]
    test_features = torch.tensor(test_dict["embeddings"], dtype=torch.float32)
    test_labels = test_dict["y_true"]

    print(train_features.shape, train_features.dtype)
    print(train_labels.shape, train_labels.dtype)
    print(test_features.shape, test_features.dtype)
    print(test_labels.shape, test_labels.dtype)

    print("Normalizing Features")
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels, test_labels = relabel(train_labels, test_labels)

    classes_train, num_classes_train = np.unique(train_labels, return_counts=True)
    classes_test, num_classes_test = np.unique(test_labels, return_counts=True)

    num_classes_train = len(num_classes_train)
    num_classes_test = len(num_classes_test)

    print(num_classes_train, num_classes_test, classes_test, classes_train)

    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    if num_classes_train != num_classes_test:
        print("Not the same number of classes in test and train")
        print(num_classes_train, num_classes_test)

    if args.use_cuda:
        print("Transfering train features to CUDA")
        train_features = train_features.cuda()
        train_labels = train_labels.cuda()

    print("Features are ready!\nStart the k-NN classification.")

    result_dict = {"k": [], "top1": [], "top5": []}
    for k in args.nb_knn:
        top1, top5, predictions, debug_dict = knn_classifier(
            train_features,
            train_labels,
            test_features,
            test_labels,
            k,
            args.temperature,
            num_classes=num_classes_train,
        )
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        result_dict["k"].append(k)
        result_dict["top1"].append(top1)
        result_dict["top5"].append(top5)

        print(predictions.shape, test_labels.numpy().shape)

        print(result_dict)
        df = pd.DataFrame.from_dict(result_dict)
        print(df)
        df.to_csv(
            os.path.join(
                args.output_dir, os.path.splitext(test_load_features)[0] + ".csv"
            ),
            index=False,
        )

        pred_df = pd.DataFrame.from_dict(
            {"y_pred": predictions, "y_true": test_labels.cpu().numpy()}
        )
        pred_df.to_csv(
            os.path.join(
                args.output_dir_pred,
                os.path.splitext(test_load_features)[0] + f"_k{k}.csv",
            ),
            index=False,
        )
