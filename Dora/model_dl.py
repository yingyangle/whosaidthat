# # coding: utf-8
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# import json
# import gensim
# from nltk import word_tokenize
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
# from matplotlib import pyplot as plt
# import seaborn as sns
# import pytorch
#
#
# dataset_name = "bang.csv"
# embed_dim = 128
#
#
# def read_files(dataset):
#     train_name = dataset + ".train"
#     test_name = dataset + ".test"
#     labels_map = json.load(open(dataset + ".labelmap", "r"))
#     train_df, test_df = pd.read_csv(train_name), pd.read_csv(test_name)
#     train_labels = np.array([labels_map[name] for name in train_df["Speaker"]])
#     test_labels = np.array([labels_map[name] for name in test_df["Speaker"]])
#     w2v_model = gensim.models.KeyedVectors.load_word2vec_format(dataset + ".w2v")
#     train_data = np.zeros([len(train_df), embed_dim])
#     test_data = np.zeros([len(test_df), embed_dim])
#     # trian
#     for idx, sentence in enumerate(train_df["Line"]):
#         try:
#             tokens = word_tokenize(sentence)
#         except TypeError:
#             tokens = ["nan"]
#         for word in tokens:
#             if word in w2v_model.vocab:
#                 train_data[idx] += w2v_model[word]
#     # test
#     for idx, sentence in enumerate(test_df["Line"]):
#         try:
#             tokens = word_tokenize(sentence)
#         except TypeError:
#             tokens = ["nan"]
#         for word in tokens:
#             if word in w2v_model.vocab:
#                 test_data[idx] += w2v_model[word]
#
#     return train_data, test_data, train_labels, test_labels
#
#
# if __name__ == "__main__":
#     train_x, test_x, train_y, test_y = read_files(dataset_name)
#     clf = LogisticRegression(solver='lbfgs', multi_class=True)
#     clf.fit(train_x, train_y)
#     pred_y = clf.predict(test_x)
#     acc = accuracy_score(test_y, pred_y)
#     p_score = precision_score(test_y, pred_y, average="macro")
#     r_score = recall_score(test_y, pred_y, average="macro")
#     f1 = f1_score(test_y, pred_y, average="macro")
#     print("accuracy: [{}]\n precision: [{}]\n recall: [{}]\n f1: [{}]".format(acc, p_score, r_score, f1))
#     sns.set()
#     f, ax = plt.subplots()
#
#     C2 = confusion_matrix(test_y, pred_y, labels=[0, 1, 2])
#     sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
#     ax.set_title('confusion matrix')  # 标题
#     ax.set_xlabel('predict')  # x轴
#     ax.set_ylabel('true')  # y轴
