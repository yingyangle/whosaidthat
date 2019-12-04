# Christine Yang
# NLP Project: whosaidthat
# getData.py
# get dialogue lines from csv files and normalize text

import os, re, nltk, math, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
# from word2number import w2n

# your_path = '/Users/Christine/Documents/cs/whosaidthat' # christine
your_path = '/Users/user/NLP Project/whosaidthat-1' # dora

# split data in with n percent for testing, rest for training
# returns 2 dfs for training and testing data
def splitData(filename, n):
    # df of all original text data
    df = pd.read_csv(filename) #, encoding='ISO-8859-1')
    x = df.iloc[:,1:2].values # lines
    y = df.iloc[:,0:1].values # speakers
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=n)
    train = pd.DataFrame({'Speaker': y_train.flatten(), 'Line': X_train.flatten()})
    test = pd.DataFrame({'Speaker': y_test.flatten(), 'Line': X_test.flatten()})
    return (train, test)

# get lines from df for a character or list of characters
def getLines(df, characters):
    if type(characters) is str: # if only looking for one character
        lines = df.loc[df.Speaker == characters].iloc[:, 1:].values
    else: # if we want lines for a list of characters
        lines = df[df['Speaker'].isin(characters)].iloc[:, 1:].values
    return lines

# get list of main characters in show
def getCast(df):
    return list(df.Speaker.unique())

# normalize data
def normalizeData(original):
    # original is a nested list of str: [[str],[str],...]
    # list to store a list of tokens for each utterance. [['token','token',...],['token','token',...]]
    list_of_tokenized_lines = []
    # tokenization
    for i in original:
        utterance = str(i[0])
        utterance = utterance.lower() # lower case everything
        utterance = text2int(utterance) # text to number: twenty-six= 26
        tokens = nltk.word_tokenize(utterance)
        list_of_tokenized_lines.append(tokens)
    return list_of_tokenized_lines

# convert all number words in utterance to actual digit numbers
def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        tens = [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"
        ]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

       # numwords["and"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10**(idx * 3 or 2), 0)

    ordinal_words = {
        'first': 1,
        'second': 2,
        'third': 3,
        'fifth': 5,
        'eighth': 8,
        'ninth': 9,
        'twelfth': 12
    }
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring


# testing
# os.chdir(your_path)


# train, test = splitData(filename, 0.2) # split train/test data
# train = getLines(train, 'Sheldon') # get Sheldon's lines in train data
# print('test data:', len(train), '\ntrain data:', len(test))

## test the characters in the original files:
# filename = 'desperate.csv'
filename = 'bang.csv'
df = pd.read_csv(filename)
cast = getCast(df)
print(cast)
