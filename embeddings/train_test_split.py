# Duanchen Liu
# coding: utf-8
import pandas as pd
import json
import random
import os


# dataset = "bang.csv"
# dataset = "simpsons.csv"
dataset = "desperate.csv"
split_ratio = 0.8


def preprocess_raw_dataset(dataset):
    raw_data = pd.read_csv(dataset)
    raw_data.fillna("nan", inplace=True)
    raw_data["Line"] = [x.lower() for x in raw_data["Line"]]
    unique_labels = set(raw_data["Speaker"])
    unique_labels = sorted(unique_labels)
    labels_int = range(len(unique_labels))
    labels_dict = dict(zip(unique_labels, labels_int))
    print(unique_labels)
    if not os.path.exists(dataset + ".labelmap"):
        json.dump(labels_dict, open(dataset + ".labelmap", "w"))
    return raw_data, labels_dict


def train_test_split(dataset, dataframe, lab_dict, ratio):
    unique_labels = list(lab_dict.keys())
    train_list = []
    test_list = []
    for label in unique_labels:
        spec_idx = set(dataframe.loc[dataframe["Speaker"] == label].index)
        samples_num = int(ratio * len(spec_idx))
        train_idx = random.sample(spec_idx, samples_num)
        temp_set = set(train_idx)
        test_idx = list(spec_idx - temp_set)
        train_list, test_list = train_list + train_idx, test_list + test_idx

    train_df = dataframe.loc[train_list]
    test_df = dataframe.loc[test_list]
    if not os.path.exists(dataset + ".train"):
        train_df.to_csv(dataset + ".train")
        test_df.to_csv(dataset + ".test")


if __name__ == "__main__":
    data, label_dict = preprocess_raw_dataset(dataset)
    train_test_split(dataset, data, label_dict, split_ratio)
    print("Finished without error.")
