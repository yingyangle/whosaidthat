# coding: utf-8
import pandas as pd
import os
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

# dataset_name = "simpsons.csv"
dataset_name = "desperate.csv"

def read_data(dataset):
    dataframe = pd.read_csv(dataset)
    dataframe.fillna("nan", inplace=True)
    texts = [line.lower() + '\n' for line in list(dataframe["Line"])]
    with open(dataset + ".temp", "w", encoding="utf-8") as f:
         f.writelines(texts)


def train_w2v(dataset):
    sentences = LineSentence(dataset + ".temp")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    if not os.path.exists(dataset + ".w2v"):
        wv_model = Word2Vec(sentences, window=5, size=128, min_count=3, workers=cpu_count(), iter=100)
        wv_model.wv.save_word2vec_format(dataset + ".w2v")
    else:
        print("word embedding already exists")
    os.remove(dataset + ".temp")


if __name__ == "__main__":
    read_data(dataset_name)
    train_w2v(dataset_name)
    print("Finished without error.")

