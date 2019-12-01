# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/
import nltk
import os, re, nltk, math, pandas as pd
from getData import getData
from getData import normalizeData
from nltk.corpus import words
from nltk.corpus import stopwords

allwords = words.words()  # all english words
stopwords = stopwords.words('english')


#############################helper functions################################
# utterance length
def utterLength(line):
    return len(line)


# average word length
def wordLength(line):
    charCount = len(''.join(line))
    wordCount = len(line)
    return charCount / wordCount


# utterance type
def utterType(line):
    questions = line.count('?')
    exclamations = line.count('!')
    ellipses = line.count('...')
    return [questions, exclamations, ellipses]


# top words

# sentiment


# diversity of vocabulary, aka type-token ratio
def typeToken(line):
    return len(set(line)) / len(line)


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
    # for num in range(10):
    #     if num in line: return 1
    for s in line:
        if s.isdigit():
            return 1
    return 0


# contains profanity
# 1 = contains profanity, 0 = doesn't contain profanity


def hasProfanity(line):
    # os.chdir('/Users/Christine/Documents/cs/whosaidthat/features') # christine
    os.chdir('/Users/user/NLP Project/whosaidthat/features')  # dora

    ein = open('profanity.txt', 'r')
    profanity = ein.read().rstrip().split('\n')
    ein.close()
    for p in profanity:
        if p in line:
            return 1
    return 0


###################################################################
# list of features for each lines
feature_list = []

# take in a line as a list of tokens
# os.chdir('/Users/Christine/Documents/cs/whosaidthat') # christine
os.chdir('/Users/user/NLP Project/whosaidthat')  # dora
data = getData('bang.csv', 'Leonard')
normalized_data = normalizeData(data)
create_features(normalized_data)
print(len(feature_list))


# print(normalized_data)
def create_features(lines):
    for i in range(len(lines)):
        line_feature_list = []
        line = lines[i]
        line_feature_list.append(utterLength(line))  #utterance len
        line_feature_list.append(wordLength(line))  #avg word len
        line_feature_list.extend(utterType(line))  # num of ? ! ...
        # top word placeholder, maybe not needed
        # sentiment placeholder, maybe not needed since we have utterType
        line_feature_list.append(typeToken(line))  # type-token ratio
        line_feature_list.append(
            neologisms(line))  # ratio of neologisms to words
        line_feature_list.append(stopwordsratio(line))  # stopwords ratio
        # POS ratio placeholder, maybe not needed
        line_feature_list.append(hasNumbers(line))  # has number=1, ow=0
        line_feature_list.append(hasProfanity(line))  # has profanity=1, ow=0

        feature_list.append(line_feature_list)


# print(normalized_data)
