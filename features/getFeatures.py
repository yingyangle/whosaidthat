# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/

import os, re, nltk, pandas as pd, numpy as np
from nltk.corpus import words
from nltk.corpus import stopwords
from getData import getData, getCast, normalizeData
from topWords import getTopWords

your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora

allwords = words.words() # all english words
stopwords = stopwords.words('english') # stop words

ein = open(your_path + '/features/profanity.txt', 'r')
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

# how many top words this line contains
def numTopWords(line, top_words):
    ans = 0
    for s in line:
        if s in top_words: ans += 1
    return ans

# contains numbers
# 1 = contains numbers, 0 = doesn't contain numbers
def hasNumbers(line):
    for s in line:
        if s.isdigit(): return 1
    return 0

# contains profanity
# 1 = contains profanity, 0 = doesn't contain profanity
def hasProfanity(line):
    for p in profanity:
        if p in line: return 1
    return 0

############################# helper functions ################################


############################# main ################################

# get features for list of lines of dialogue
# takes in list of lists of tokens, returns list of feature lists
def getFeatures(lines, top_words):
    features = [] # full list of features for each line
    for line in lines: # for each line
        line_features = np.array([ # features for this line
            len(line), # utterance len
            len(''.join(line)) / len(line), # avg word len
            numTopWords(line, top_words), # number of top words in this line
            len(set(line)) / len(line), # type-token ratio
            neologisms(line), # ratio of neologisms to words
            stopwordsratio(line), # stopwords ratio
            hasNumbers(line), # has number=1, ow=0
            hasProfanity(line) # has profanity=1, ow=0
        ])
        np.concatenate((line_features, utterType(line))) # num of ? ! ...)
        features.append(line_features) # add this line's features to full list
    return features

# create features dataset for given show and character, write to .csv, return df
# label lines 1 if spoken by the focus character, 0 otherwise
def createDataset(csv_filename, character):
    ones_lines = getData(csv_filename, character) # lines for focus character
    ones_lines = normalizeData(ones_lines)
    top_words = getTopWords(ones_lines, 20) # 20 most frequent words for this character
    ones_feats = getFeatures(ones_lines, top_words) # features for focus character's lines
    ones = np.ones_like(ones_lines) # list of 1's labels

    cast = getCast(csv_filename) # list of main characters in show
    cast.remove(character) # remove focus character from character list
    zeros_lines = getData(csv_filename, cast) # lines for remaining characters
    zeros_lines = normalizeData(zeros_lines)
    zeros_feats = getFeatures(zeros_lines, top_words) # features for remainin characters' lines
    zeros = np.zeros_like(zeros_lines) # list of 0's labels

    labels = list(np.concatenate((ones, zeros), axis=0).flatten())
    feats = list(np.concatenate((ones_feats, zeros_feats), axis=0))
    print('Total Datapoints:', len(labels))
    print(character, 'Datapoints:', len(ones))
    data = {'Label': labels, 'Features': feats}
    df = pd.DataFrame(data) # dataframe of features for each line and 1/0 label
    df.to_csv(csv_filename[:-4] + character + ".csv", index=False, encoding='utf-8-sig') # save to .csv
    return df

############################# main ################################


############################# execute ################################

os.chdir(your_path)
bangSheldon = createDataset('bang.csv', 'Sheldon')
