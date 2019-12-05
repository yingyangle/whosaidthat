# Christine Yang
# NLP Project: whosaidthat
# runModels.py
# run logistic regression and random forest models on feature datasets

# 1. run getData.py
# 2. run getFeatures.py
# 3. run runModels.py (this file)

import os, re, nltk, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from os.path import join

your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora
# your_path = "/Users/julianafakhoury/Documents/BC/nlp_project/newnewnew/whosaidthat" # juliana

os.chdir(your_path+'/features')
os.chdir(your_path)

# get X_train, X_test, y_train, y_test
def getData(train, test):
    # training data
    X_train = np.array([x for x in train.values[:,1]])
    y_train = train.values[:,0]
    y_train = y_train.astype('int')
    # testing data
    X_test = np.array([x for x in test.values[:,1]])
    y_test = test.values[:,0]
    y_test = y_test.astype('int')
    return (X_train, X_test, y_train, y_test)

# run logistic regression and random forest models
def runModels(X_train, X_test, y_train, y_test):
    # logistic regression
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(X_train, y_train)
    # log_score = log.score(X_test, y_test)
    log_predict = log.predict(X_train)
    label_vals = list(np.unique(y_test))
    log_score = classification_report(y_train, log_predict, labels=label_vals)

    # random forest
    rf = RandomForestClassifier(n_estimators = 400, max_features = 3, oob_score = True)
    rf.fit(X_train, y_train.ravel())
    # rf_score = rf.score(X_test, y_test)
    rf_predict = rf.predict(X_train)
    rf_score = classification_report(y_train, rf_predict, labels=label_vals)
    return [log_score, rf_score]


### execute ###

shows = ['bang', 'simpsons', 'desperate']
shows = ['bang']

for show in shows: # for each show
    print(show, '{0:~^40}'.format(''))
    # get train/test data
    train = pd.read_pickle(join(your_path,'datasets/features_data/'+show+'Train.pkl'))
    test = pd.read_pickle(join(your_path,'datasets/features_data/'+show+'Test.pkl'))
    X_train, X_test, y_train, y_test = getData(train, test)
    # train and test models, get accuracy metrics
    scores = runModels(X_train, X_test, y_train, y_test)
    print('\tLogistic Regression:\n', scores[0])
    print('\tRandom Forest:\n', scores[1])
