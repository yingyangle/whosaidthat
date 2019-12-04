#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:41:42 2019

@author: julianafakhoury
"""
import gensim
import os, re, nltk, math, pandas as pd, numpy as np
from nltk.corpus import stopwords
import numpy as np
import scipy as sp
import re
from sklearn.cluster import KMeans
# import getDataEmbeddings
from getDataEmbeddings import splitData, getLines, getCast, normalizeData
import urllib
import sklearn.linear_model as skl
import sklearn.ensemble as ske
import seaborn as sns
import sklearn.tree as skt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

def make_embeddings(filename,character,bigmodel):
    
    #prepare lines 
    df = pd.read_csv(filename)
   
    #get lines and vectors for char
    main_lines = getLines(df, character)
    main_lines = normalizeData(main_lines)
    
    main_x = []  # this list will contain one 300-dimensional vector per line
    
    for h in main_lines:
        totvec = np.zeros(300)
        for w in h:
            if w.lower() in bigmodel:
                totvec = totvec + bigmodel[w.lower()]
        main_x.append(totvec)
    
    main_x = pd.DataFrame(main_x)
    main_y = pd.DataFrame(np.ones(len(main_lines)))

    #get lines and vectors for all other chars
    cast = getCast(df)
    cast.remove(character)
    #cast = cast.remove(character)
    
    
    rest_lines = getLines(df, cast)
    rest_lines = normalizeData(rest_lines)
    
    rest_x = []  # this list will contain one 300-dimensional vector per line
    
    for h in rest_lines:
        totvec = np.zeros(300)
        for w in h:
            if w.lower() in bigmodel:
                totvec = totvec + bigmodel[w.lower()]
        rest_x.append(totvec)
        
    rest_x = pd.DataFrame(rest_x)
    rest_y = pd.DataFrame(np.zeros(len(rest_lines)))
    
    #concat datasets 
    x = pd.concat([main_x,rest_x])
    y = pd.concat([main_y,rest_y])
 
    return x, y

# path = "/Users/julianafakhoury/Documents/BC/nlp_project/new/GoogleNews-vectors-negative300-SLIM.bin" #juliana
path = "/Users/user/NLP Project/GoogleNews-vectors-negative300-SLIM.bin" #dora

bigmodel = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
x, y = make_embeddings('bang.csv','Sheldon',bigmodel )

x = x.values
y = y.values

X_main, X_validation, y_main, y_validation = train_test_split(x,y,test_size=0.20)


#Logistic Regression 
lor = skl.LogisticRegression()
lor.fit(X_main, y_main.ravel())


#Random Forest, 400 and 3 picked in the optimization code after line 63 and manually put here 
rf = ske.RandomForestClassifier(n_estimators = 400, max_features = 17, oob_score = True)
rf.fit(X_main, y_main.ravel())


#Visualizing 
print("Logistic Regression Score ", lor.score(X_validation, y_validation))
print("Random Forest Score ", rf.score(X_validation, y_validation))