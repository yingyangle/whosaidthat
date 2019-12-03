# Christine Yang
# NLP Project: whosaidthat
# logreg.py
# run logistic regression model

import os, re, nltk, pandas as pd, numpy as np
from getFeatures import createDataset
from sklearn.linear_model import LogisticRegression

# your_path = '/Users/Christine/cs/whosaidthat' # christine
your_path = '/Users/user/NLP Project/whosaidthat' # dora
os.chdir(your_path+'/features')
os.chdir(your_path)

train = pd.read_pickle('datasets/bangSheldonTrain.pkl')
test = pd.read_pickle('datasets/bangSheldonTest.pkl')

# training data
X_train = np.array([x for x in train.values[:,1]])
y_train = train.values[:,0]
y_train = y_train.astype('int')

# testing data
X_test = np.array([x for x in test.values[:,1]])
y_test = train.values[:,0]
y_test = y_test.astype('int')

log = LogisticRegression(solver='liblinear', multi_class='ovr')
log.fit(X_train, y_train)
predictions = log.predict(X_test)
score = log.score(X_test, y_test)
print('Accuracy:', score)
print(train)
print(test)