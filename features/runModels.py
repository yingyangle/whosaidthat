# Christine Yang
# NLP Project: whosaidthat
# runModels.py
# run logistic regression and random forest models on feature datasets
# takes about ~30 min. for each show

# 1. run getData.py
# 2. run getFeatures.py
# 3. run runModels.py (this file)

import os, re, nltk, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from os.path import join
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
import seaborn as sns

# your_path = '/Users/Christine/cs/whosaidthat' # christine
your_path = '/Users/user/NLP Project/whosaidthat-3' # dora
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
    # log_predict = log.predict(X_train) # christine
    log_predict = log.predict(X_test) # dora
    label_vals = list(np.unique(y_test))
    # log_score = classification_report(y_train, log_predict, labels=label_vals) #christine
    log_score = classification_report(y_test, log_predict, labels=label_vals) # dora

    # random forest
    rf = RandomForestClassifier(n_estimators = 400, max_features = 3, oob_score = True)
    rf.fit(X_train, y_train.ravel())
    # rf_score = rf.score(X_test, y_test)
    # rf_predict = rf.predict(X_train) #christine
    rf_predict = rf.predict(X_test) # dora
    # rf_score = classification_report(y_train, rf_predict, labels=label_vals) #christine
    rf_score = classification_report(y_test, rf_predict, labels=label_vals) #dora



    #Neuronet
    clf = MLPClassifier(batch_size=8, learning_rate="adaptive", solver="sgd", max_iter=100, hidden_layer_sizes=200 )
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)
    nn_score = classification_report(y_test, pred_y, labels=label_vals) #dora

     #confusion matrix
    # sns.set()
    # f, ax = plt.subplots()
    # C2 = confusion_matrix(y_test, rf_predict, labels=label_vals)
    # sns.heatmap(C2,annot=True, linewidths=.5, fmt="d", ax=ax)
    # ax.set_title('Confusion Matrix')
    # ax.set_xlabel('predict')
    # ax.set_ylabel('true')
    # plt.show()


    return [log_score, rf_score, nn_score]


### execute ###

shows = ['bang', 'simpsons', 'desperate']
# shows = ['desperate']
# shows = ['bang']
shows = ['simpsons']
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
    print('\tMulti-layer Perceptron:\n', scores[2])

   
