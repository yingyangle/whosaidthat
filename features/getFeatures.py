# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/

import os, re, nltk, pandas as pd, numpy as np, pickle
from nltk.corpus import words
from nltk.corpus import stopwords
from getData import splitData, getLines, getCast, normalizeData
from topWords import getTopWords

# your_path = '/Users/Christine/cs/whosaidthat' # christine
your_path = '/Users/user/NLP Project/whosaidthat-1'  # dora

allwords = words.words()  # all english words
stopwords = stopwords.words('english')  # stop words

ein = open(your_path + '/features/profanity.txt', 'r')
profanity = ein.read().rstrip().split('\n')  # profanity words
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
            hasProfanity(line)]) # has profanity=1, ow=0
        line_features = np.concatenate((line_features,utterType(line)))  # num of ? ! ...)
        features.append(line_features) # add this line's features to full list
    return features

# convert speaker/line df to features/label df for given character
# label is 1 if line spoken by the focus character, 0 otherwise
def convertToFeatures(df, character, top_words):
    # ones_lines = getLines(df, character)[:10]  # lines for focus character, only 10 lines
    ones_lines = getLines(df, character) # lines for focus character, full
    ones_lines = normalizeData(ones_lines)
    ones_feats = getFeatures(ones_lines, top_words) # features for focus character's lines
    ones = np.ones_like(ones_lines) # list of 1's labels

    cast = getCast(df) # list of main characters in show
    cast.remove(character) # remove focus character from character list
    # zeros_lines = getLines(df, cast)[:10] # lines for remaining characters, only 10 lines
    zeros_lines = getLines(df, cast) # lines for remaining characters, full
    zeros_lines = normalizeData(zeros_lines)
    zeros_feats = getFeatures(zeros_lines, top_words) # features for remainin characters' lines
    zeros = np.zeros_like(zeros_lines) # list of 0's labels

    labels = list(np.concatenate((ones, zeros), axis=0).flatten())
    feats = list(np.concatenate((ones_feats, zeros_feats), axis=0))
    print(character, 'Datapoints:', len(ones), ', Total Datapoints:', len(labels))
    data = {'Label': labels, 'Features': feats}
    df = pd.DataFrame(data) # dataframe of features for each line and 1/0 label
    return df

# create train and test dataset for given show and character
# write resulting datsets to .pkl and return dfs
def createDataset(filename, character):
    train, test = splitData(filename, 0.2)  # split 80/20 for training/testing data
    char_lines = getLines(train, character)  # get character's lines
    top_words = getTopWords(char_lines, 20)  # 20 most frequent words for character
    train = convertToFeatures(train, character, top_words)
    test = convertToFeatures(test, character, top_words)
    # save to .csv
    name = filename[:-4] + character
    train.to_pickle('datasets/' + name + 'Train.pkl')
    test.to_pickle('datasets/' + name + 'Test.pkl')
    return (train, test)

############################# main ################################


############################# execute ################################

os.chdir(your_path)

# Big Bang Theory
# # create dataset for Big Bang and Sheldon - done！
# train, test = createDataset('bang.csv', 'Sheldon')
# # create dataset for Big Bang and Penny - done!
# train, test = createDataset('bang.csv', 'Penny')
# # create dataset for Big Bang and Leonard - done！
# train, test = createDataset('bang.csv', 'Leonard')
# # # create dataset for Big Bang and Raj - done！
# train, test = createDataset('bang.csv', 'Raj')
# # # create dataset for Big Bang and Amy - done！
# train, test = createDataset('bang.csv', 'Amy')
# # create dataset for Big Bang and Bernadette - done！
# train, test = createDataset('bang.csv', 'Bernadette')
# # create dataset for Big Bang and Howard - done！
# train, test = createDataset('bang.csv', 'Howard')

# The Simpsons
# create dataset for Simpsons and Homer - done
# train, test = createDataset('simpsons.csv', 'Homer')
# create dataset for Simpsons and Marge - done
# train, test = createDataset('simpsons.csv', 'Marge')
# # create dataset for Simpsons and Lisa - done
# train, test = createDataset('simpsons.csv', 'Lisa')
# # create dataset for Simpsons and Bart - done
# train, test = createDataset('simpsons.csv', 'Bart')
# # create dataset for Simpsons and Ned Flanders - done
# train, test = createDataset('simpsons.csv', 'Ned Flanders')

# Desperate Housewives
# create dataset for desperate and Susan - done
# train, test = createDataset('desperate.csv', 'Susan')
