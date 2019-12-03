# Christine Yang
# NLP Project: whosaidthat
# logreg.py
# run logistic regression model

import os, re, nltk, math, pandas as pd, numpy as np
from getData import splitData, getData, normalizeData

train, test = splitData(filename, 0.2)
train = getData(train, 'Sheldon')
test = getData(test, 'Sheldon')