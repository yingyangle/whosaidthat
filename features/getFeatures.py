# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# create features/label dataset for each show from normalized text datasets

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/

import os, re, nltk, pandas as pd, numpy as np, time
from nltk.corpus import words, stopwords
from os.path import join
from getData import getCast
from topWords import getEachTopWords
from textblob import TextBlob

# your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora
your_path = "/Users/julianafakhoury/Documents/BC/nlp_project/recent/whosaidthat" # juliana

start_time = time.time()
allwords = words.words()  # all english words
stopwords = stopwords.words('english')  # stop words

ein = open(your_path + '/features/profanity.txt', 'r')
profanity = ein.read().rstrip().split('\n')  # profanity words
ein.close()


############################# extract features ################################

# ratio of stop words to all words in utterance
def stopwordsRatio(line):
    stops = [x for x in line if x in stopwords]
    return len(stops) / len(line)

# ratio of neologisms to words in utterance
def neologismsRatio(line):
    neos = [x for x in line if x not in allwords]
    return len(neos) / len(line)

# how many numbers this line contains
def numberCount(line):
    nums = [x for x in line if x.isdigit()]
    return len(nums)

# how many profanity words this line contains
def profanityCount(line):
    profs = [x for x in line if x in profanity]
    return len(profs)

# how many of each utterance type are in this line
def utterTypeCount(line):
    questions = line.count('?')
    exclamations = line.count('!')
    ellipses = line.count('...')
    return [questions, exclamations, ellipses]

# how many words in this line are in each of the top words lists for the main characters
def topWordsCount(line, eachTopWords):
    topCounts = [] # counts for words in each character's top words list
    for topWords in eachTopWords: # for each character's top words list
        # get number of words in this line that are in topWords
        topCounts.append(len([x for x in line if x in topWords]))
    return topCounts

#sentiment 
def subjectivity(line):
    #empty string
    line_str = ""
    #transform into full sentence 
    for i in line:
        line_str = line_str + i + " "
    #apply TextBlob
    sub = TextBlob(line_str)
    return sub.sentiment.subjectivity

#sentiment 
def polarity(line):
    #empty string
    line_str = ""
    #transform into full sentence 
    for i in line:
        line_str = line_str + i + " "
    #apply TextBlob  
    pol = TextBlob(line_str)
    return pol.sentiment.polarity
    

############################# extract features ################################


############################# create dataset ################################

# get features for list of lines of dialogue
# takes in list of lists of tokens, returns list of feature lists
def getFeatures(lines, eachTopWords):
    feats = [] # full list of features for each line
    for line in lines: # for each line
        lineFeats = np.array([ # get features for this line
            len(line), # utterance len
            len(''.join(line)) / len(line), # avg word len
            len(set(line)) / len(line), # type-token ratio
            stopwordsRatio(line),
            neologismsRatio(line),
            numberCount(line),
            profanityCount(line),
            subjectivity(line),
            polarity(line)])
        lineFeats = np.concatenate((lineFeats, utterTypeCount(line)))
        lineFeats = np.concatenate((lineFeats, topWordsCount(line, eachTopWords)))
        feats.append(lineFeats) # add this line's features to full feats list
    return feats

# convert speaker/line df to features/label df
# label characters 0 through N for N-1 main characters
def convertToFeatures(df, eachTopWords, writeFlag):
    # convert speakers to number labels
    speaker_vals = getCast(df)
    label_vals = list(range(len(speaker_vals)))
    labelsDict = {speaker_vals[i]:label_vals[i] for i in range(len(speaker_vals))}
    if writeFlag is 1: # write labelsDict to .txt so we know who's who
        aus = open('datasets/labels_dictionary.txt', 'a')
        aus.write(str(labelsDict)+'\n\n')
        aus.close()
    speakers = list(df['Speaker'])#[:10]
    labels = [labelsDict[x] for x in speakers]
    # convert lines to features
    lines = list(df['Line'])#[:10]
    feats = getFeatures(lines, eachTopWords)
    # save and return new df
    # print(len(feats), len(labels)) # testing
    data = {'Label': labels, 'Features': feats}
    new_df = pd.DataFrame(data)
    return new_df

# create train and test dataset for given show and character
# write resulting datsets to .pkl and return dfs
def createDataset(show):
    # load normalized tokenized data
    train = pd.read_pickle(join(your_path,'datasets/norm_text_data/'+show+'Train.pkl'))
    test = pd.read_pickle(join(your_path, 'datasets/norm_text_data/'+show+'Test.pkl'))
    # list of 20 most frequent words for each character
    eachTopWords = getEachTopWords(show)
    # convert dataset to labels/features
    train = convertToFeatures(train, eachTopWords, 1)
    test = convertToFeatures(test, eachTopWords, 0)
    # save to pickle
    train.to_pickle('datasets/features_data/' + show + 'Train.pkl')
    test.to_pickle('datasets/features_data/' + show + 'Test.pkl')
    # print progress
    print('Finished creating dataset for', show, '!', time.time()-start_time)
    return (train, test)

############################# create dataset ################################


################################ execute ###################################

os.chdir(your_path)
createDataset('bang')
createDataset('simpsons')
createDataset('desperate')

# pd.read_pickle(join(your_path, 'datasets/features_data/bangTest.pkl'))
