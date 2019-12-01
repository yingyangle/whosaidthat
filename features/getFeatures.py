# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/
import os, re, nltk, math, pandas as pd
from getData import getData
from getData import normalizeData
from nltk.corpus import words
from nltk.corpus import stopwords

your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora

allwords = words.words()  # all english words
stopwords = stopwords.words('english') # stop words
os.chdir(your_path + '/features') 
ein = open('profanity.txt', 'r')
profanity = ein.read().rstrip().split('\n') # profanity words
ein.close()


############################# helper functions ################################

# utterance type
def utterType(line):
    questions = line.count('?')
    exclamations = line.count('!')
    ellipses = line.count('...')
    return [questions, exclamations, ellipses]

# top words

# sentiment

# ratio of neologisms to words in utterance
def neologisms(line):
    neos = [x for x in line if x not in allwords]
    return len(neos) / len(line)

# ratio of stop words to words in utterance
def stopwordsratio(line):
    stops = [x for x in line if x in stopwords]
    return len(stops) / len(line)

# POS ratio

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
def get_features(lines):
    features = [] # full list of features for each line
    for i in range(len(lines)): # for each line
        line_features = [] # features for this line
        line = lines[i] # this line
        line_features.append(len(line))  # utterance len
        line_features.append(len(''.join(line))/len(line))  # avg word len
        line_features.extend(utterType(line))  # num of ? ! ...
        # top word placeholder, maybe not needed
        # sentiment placeholder, maybe not needed since we have utterType
        line_features.append(len(set(line))/len(line))  # type-token ratio
        line_features.append(
            neologisms(line))  # ratio of neologisms to words
        line_features.append(stopwordsratio(line))  # stopwords ratio
        # POS ratio placeholder, maybe not needed
        line_features.append(hasNumbers(line))  # has number=1, ow=0
        line_features.append(hasProfanity(line))  # has profanity=1, ow=0
        features.append(line_features) # add this line's features to full list
    return features


############################# execute ################################

data = getData('bang.csv', 'Leonard') # get data
normalized_data = normalizeData(data) # normalize data
features = get_features(normalized_data) # get features

print(len(features)) 
# print(normalized_data)