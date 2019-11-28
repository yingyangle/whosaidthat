# Christine Yang
# NLP Project: whosaidthat
# getData.py
# get dialogue lines from csv files and normalize

import os, pandas as pd
import nltk
import math
import re

# Christine:
# os.chdir('/Users/Christine/Documents/cs/whosaidthat')
# Dora:
os.chdir('/Users/user/NLP Project/whosaidthat')

# os.chdir('/Users/user/NLP Project/whosaidthat/BangNewestData')


# get lines from filename for a character or list of characters
def getData(filename, characters):
    df = pd.read_csv(filename)  # df = dataframe
    if type(characters) is str:  # if only looking for one character
        lines = df.loc[df.Speaker == characters].iloc[:, 1:].values
    else:  # if we want lines for a list of characters
        lines = df[df['Speaker'].isin(characters)].iloc[:, 1:].values
    # print(lines)
    return lines


# normalize data
def normalizeData(original):
    # original is a nested list of str: [[str],[str],...]
    # list to store a list of tokens for each utterance. [['token','token',...],['token','token',...]]
    list_of_tokenized_lines = []
    ################## Tokenization ############################
    for i in original:
        utterance = str(i[0])
        utterance = utterance.lower()  # lower case everything
        tokens = nltk.word_tokenize(utterance)
        list_of_tokenized_lines.append(tokens)
    # print(list_of_tokenized_lines)
    return list_of_tokenized_lines


# testing
# getData('bang.csv', 'Sheldon')
# getData('bang.csv', 'Leonard')
##############data is a nested list of str#########################
data = getData('bang.csv', ['Sheldon', 'Leonard'])
normalizeData(data)

# Dora testing
# getData('bangSevenChars.csv', 'Sheldon')
# getData('bangSevenChars.csv', ['Sheldon', 'Leonard'])
