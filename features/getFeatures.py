# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/

import os, re, nltk, math, pandas as pd, numpy as np
from nltk.corpus import words
from nltk.corpus import stopwords
from getData import getData, getCast, normalizeData

your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora

allwords = words.words()  # all english words
stopwords = stopwords.words('english') # stop words

ein = open(your_path+'/features/profanity.txt', 'r')
profanity = ein.read().rstrip().split('\n') # profanity words
ein.close()


############################# helper functions ################################

# utterance type
def utterType(line):
    questions = line.count('?')
    exclamations = line.count('!')
    ellipses = line.count('...')
    return [questions, exclamations, ellipses]

# ratio of neologisms to words in utterance
def neologisms(line):
    neos = [x for x in line if x not in allwords]
    return len(neos) / len(line)

# ratio of stop words to words in utterance
def stopwordsratio(line):
    stops = [x for x in line if x in stopwords]
    return len(stops) / len(line)

# contains numbers
# 1 = contains numbers, 0 = doesn't contain numbers
def hasNumbers(line):
    for s in line:
        if s.isdigit():
            return 1
    return 0

# contains profanity
# 1 = contains profanity, 0 = doesn't contain profanity
def hasProfanity(line):
    for p in profanity:
        if p in line:
            return 1
    return 0

############################# helper functions ################################


############################# main ################################

# get features for list of lines of dialogue
# takes in list of token lists, returns list of feature lists
def getFeatures(lines):
    features = []  # full list of features for each line
    for i in range(len(lines)):  # for each line
        line = lines[i]  # this line
        line_features = []  # features for this line
        line_features.append(len(line))  # utterance len
        line_features.append(len(''.join(line)) / len(line))  # avg word len
        line_features.extend(utterType(line))  # num of ? ! ...
        # top word placeholder, maybe not needed
        line_features.append(len(set(line)) / len(line))  # type-token ratio
        line_features.append(neologisms(line))  # ratio of neologisms to words
        line_features.append(stopwordsratio(line))  # stopwords ratio
        line_features.append(hasNumbers(line))  # has number=1, ow=0
        line_features.append(hasProfanity(line))  # has profanity=1, ow=0
        features.append(line_features)  # add this line's features to full list
    return features

# create features dataset for given show and character, write to .csv, return df
# label lines 1 if spoken by the focus character, 0 otherwise
def createDataset(csv_filename, character):
    ones_lines = getData(csv_filename, character) # lines for focus character
    ones_lines = normalizeData(ones_lines)
    ones_feats = getFeatures(ones_lines) # features for focus character's lines
    ones = np.ones_like(ones_lines) # list of 1's labels

    cast = getCast(csv_filename) # list of main characters in show
    cast.remove(character) # remove focus character from character list
    zeros_lines = getData(csv_filename, cast) # lines for remaining characters
    zeros_lines = normalizeData(zeros_lines)
    zeros_feats = getFeatures(zeros_lines) # features for remainin characters' lines
    zeros = np.zeros_like(zeros_lines) # list of 0's labels

    labels = list(np.concatenate((ones, zeros), axis=0).flatten())
    lines = list(np.concatenate((ones_lines, zeros_lines), axis=0).flatten())
    data = {'Label':labels, 'Line':lines}
    df = pd.DataFrame(data) # dataframe of features for each line and 1/0 label
    df.to_csv(csv_filename[:-4]+character+".csv", index=False, encoding='utf-8-sig') # save to .csv
    return df


############################# main ################################


############################# execute ################################

os.chdir(your_path)
bangSheldon = createDataset('bang.csv', 'Sheldon')
