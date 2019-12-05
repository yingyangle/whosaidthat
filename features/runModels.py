# Christine Yang
# NLP Project: whosaidthat
# logreg.py
# run logistic regression model

import os, re, nltk, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
# import sklearn.linear_model as skl
import sklearn.ensemble as ske
# import sklearn.tree as skt
from getData import getCast
from sklearn.metrics import classification_report


your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora
# your_path = "/Users/julianafakhoury/Documents/BC/nlp_project/newnewnew/whosaidthat" #juliana

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
    log_score = classification_report(y_train, log_predict, labels=[1,0])

    # random forest
    rf = ske.RandomForestClassifier(n_estimators = 400, max_features = 3, oob_score = True)
    rf.fit(X_train, y_train.ravel())
    # rf_score = rf.score(X_test, y_test)
    rf_predict = rf.predict(X_train)
    rf_score = classification_report(y_train, rf_predict, labels=[1,0])
    return [log_score, rf_score]

# execute
show = 'bang'
character = 'Penny'
for show in ['bang', 'simpsons', 'desperate']: # for each show
    print(show, '{0:~^20}'.format(''))
    characters = getCast(pd.read_csv(show+'.csv')) # get main characters
    for character in characters: # for each character
        print(show, '-', character, '{0:*^10}'.format(''))
        train = pd.read_pickle('datasets/'+show+character+'Train.pkl')
        test = pd.read_pickle('datasets/'+show+character+'Test.pkl')
        try:
            X_train, X_test, y_train, y_test = getData(train, test)
        except:
            print('FAILED', character)
            continue
        scores = runModels(X_train, X_test, y_train, y_test)
        print('\tLogistic Regression:\n', scores[0])
        print('\tRandom Forest:\n', scores[1])
