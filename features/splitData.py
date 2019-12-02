# Christine Yang
# NLP Project: whosaidthat
# splitData.py
# split data into training and testing datasets

import pandas as pd, os
from sklearn.model_selection import train_test_split

# split data in with n percent for testing, rest for training
def splitData(filename, n):
    df = pd.read_csv(filename) # df of all original text data
    x = df.iloc[:,1:2].values
    y = df.iloc[:,0:1].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=n)
    return (X_train, X_test, y_train, y_test)

filename = 'bang.csv'
X_train, X_test, y_train, y_test = splitData(filename, 0.2)
