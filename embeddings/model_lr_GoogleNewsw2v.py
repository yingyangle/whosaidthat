# Duanchen Liu 
# coding: utf-8
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import gensim
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
import seaborn as sns


dataset_name = "bang.csv"
# dataset_name = "simpsons.csv"
# dataset_name = "desperate.csv"
embed_dim = 128

model_path = "/Users/user/NLP Project/GoogleNews-vectors-negative300-SLIM.bin" # dora
bigmodel = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
print("big model built")


def read_files(dataset):
    train_name = dataset + ".train"
    test_name = dataset + ".test"
    labels_map = json.load(open(dataset + ".labelmap", "r"))
    train_df, test_df = pd.read_csv(train_name), pd.read_csv(test_name)
    train_labels = np.array([labels_map[name] for name in train_df["Speaker"]])
    test_labels = np.array([labels_map[name] for name in test_df["Speaker"]])
    bigmodel = gensim.models.KeyedVectors.load_word2vec_format(dataset + ".w2v")
    train_data = np.zeros([len(train_df), embed_dim])
    test_data = np.zeros([len(test_df), embed_dim])
    # trian
    for idx, sentence in enumerate(train_df["Line"]):
        try:
            tokens = word_tokenize(sentence)
        except TypeError:
            tokens = ["nan"]
        for word in tokens:
            if word in bigmodel.vocab:
                train_data[idx] += bigmodel[word]
    # test
    for idx, sentence in enumerate(test_df["Line"]):
        try:
            tokens = word_tokenize(sentence)
        except TypeError:
            tokens = ["nan"]
        for word in tokens:
            if word in bigmodel.vocab:
                test_data[idx] += bigmodel[word]

    unique_label = list(labels_map.keys()).sort()
    return train_data, test_data, train_labels, test_labels, unique_label


if __name__ == "__main__":
    train_x, test_x, train_y, test_y, labels = read_files(dataset_name)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    p_score = precision_score(test_y, pred_y, average="macro")
    r_score = recall_score(test_y, pred_y, average="macro")
    f1 = f1_score(test_y, pred_y, average="macro")
    print("accuracy: [{}]\n precision: [{}]\n recall: [{}]\n f1: [{}]".format(acc, p_score, r_score, f1))
    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(test_y, pred_y, labels=labels)
    # sns.heatmap(C2, annot=True, ax=ax)
    sns.heatmap(C2,annot=True, linewidths=.5, fmt="d", ax=ax)
    ax.set_title('Confusion Matrix from Logistic Regression for Desperate Housewives')
    ax.set_xlabel('Predict')
    ax.set_ylabel('True')
    plt.show()

