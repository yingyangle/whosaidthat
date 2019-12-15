#!/Users/Christine/anaconda3/bin/python
# Christine Yang
# NLP Project: whosaidthat
# runModels.py
# run logistic regression and random forest models on feature datasets
# takes about ~30 min. for each show

# 1. run getData.py
# 2. run getFeatures.py
# 3. run runModels.py (this file)

import os, re, nltk, pandas as pd, numpy as np, warnings
import matplotlib, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from os.path import join
matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")

your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat-3' # dora
# your_path = "/Users/julianafakhoury/Documents/BC/nlp_project/newnewnew/whosaidthat" # juliana


bangChars = ['Amy', 'Bernadette', 'Howard', 'Leonard', 'Penny', 'Raj', 'Sheldon']
simpsChars = ['Bart', 'Homer', 'Lisa', 'Marge', 'Ned Flanders']
despChars = ['Bree', 'Gabrielle', 'Lynette', 'Susan']
charDicts = {'bang':bangChars, 'simpsons':simpsChars, 'desperate':despChars}
showNames = {'bang':'Big Bang Theory', 'simpsons':'Simpsons', 'desperate':'Desperate Housewives'}
modelNames = {0:'Logistic Regression', 1:'Random Forest', 2:'Naive Bayes', 3:'Neural Net'}

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

# run models and print accuracy metrics
def runModels(X_train, X_test, y_train, y_test, show):
    chars = charDicts[show]
    label_vals = list(np.unique(y_test))
    
    # logistic regression
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(X_train, y_train)
    # log_score = log.score(X_test, y_test)
    log_predict = log.predict(X_test)
    log_score= classification_report(y_test, log_predict, labels=chars)
    print('\tLogistic Regression:\n', log_score)

    # random forest
    rf = RandomForestClassifier(n_estimators = 400, max_features = 3, oob_score = True)
    rf.fit(X_train, y_train.ravel())
    # rf_score = rf.score(X_test, y_test)
    rf_predict = rf.predict(X_test)
    rf_score = classification_report(y_test, rf_predict, labels=chars)
    print('\tRandom Forest:\n', rf_score)

    # naive bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # gnb_score = gnb.score(X_test, y_test)
    gnb_predict = gnb.predict(X_test)
    gnb_score = classification_report(y_test, gnb_predict, labels=chars)
    print('\tNaive Bayes:\n', gnb_score)

    # neural net
    nn = MLPClassifier(batch_size=8, learning_rate="adaptive", solver="sgd", max_iter=100, hidden_layer_sizes=200 )
    nn.fit(X_train, y_train)
    nn_predict = nn.predict(X_test)
    nn_score = classification_report(y_test, nn_predict, labels=chars)
    print('\tNeural Net:\n', nn_score)    

    return 

def getBaselines(X_train, X_test, y_train, y_test):
    chars = charDicts[show]
    
    # random baseline
    random = DummyClassifier(strategy='uniform')
    random.fit(X_train, y_train) 
    random_predict = random.predict(X_test)
    # random_score = classification_report(y_test, random_predict, labels=chars)
    random_score = random.score(X_test, y_test)
    print('\tRandom Baseline:\n', random_score)    
    
    # majority baseline
    majority = DummyClassifier(strategy='most_frequent')
    majority.fit(X_train, y_train)
    majority_predict = majority.predict(X_test)
    # majority_score = classification_report(y_test, majority_predict, labels=chars)
    majority_score = majority.score(X_test, y_test)
    print('\tMajority Baseline:\n', majority_score)    
    return

# save confusion matrix heatmap as png
def getConfusionMatrix(show, y_actual, y_predict, modelName=''):
    label_vals = charDicts[show]
    sns.set()
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_actual, y_predict, target_names=label_vals)
    sns.heatmap(cm, annot=True, linewidths=.5, fmt="d", ax=ax)
    ax.set_title(showNames[show]+': '+modelName, fontsize=30)
    ax.set_xlabel('predict', fontsize=23)
    ax.set_ylabel('actual', fontsize=23)
    plt.show()
    plt.savefig(show+'Matrix.png', dpi=200)
    return


### execute ###

os.chdir(your_path+'/features')
os.chdir(your_path)

shows = ['bang', 'simpsons', 'desperate']
# shows = ['desperate']
 
for show in shows: # for each show
    print('\n\n', show, '{0:#^40}\n'.format(''))
    # get train/test data
    train = pd.read_pickle(join(your_path, 'datasets/features_data/'+show+'Train.pkl'))
    test = pd.read_pickle(join(your_path, 'datasets/features_data/'+show+'Test.pkl'))
    X_train, X_test, y_train, y_test = getData(train, test)
    chars = charDicts[show]
    # train and test models, get accuracy metrics
    scores = runModels(X_train, X_test, y_train, y_test, show)
    # train and test baseline models, get accuracy metrics
    getBaselines(X_train, X_test, y_train, y_test)
    
    
