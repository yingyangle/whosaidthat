# topWords.py
# get the top N most frequent words in a dataset of normalized text data

import os, re, nltk, pandas as pd
from nltk.corpus import stopwords
from itertools import chain
from os.path import join
from getData import getLines, getCast


your_path = '/Users/Christine/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora
# your_path = "/Users/julianafakhoury/Documents/BC/nlp_project/newnewnew/whosaidthat" #juliana

# list of stopwords to exclude when getting most frequent words list
stoplist = stopwords.words('english')
more = [',', '.', '...', '“', '”', ';', '?', '!', '-', '--', '’', "n't", "'s", \
        "'m", "'re", "'ll", "'ve", '``', "''", "''", "'", "'d", 'u', \
        'would', 'could', 'yeah', 'okay', 'get', 'well', 'wi', 'wa', 'know', \
        'right', 'want', '1', 'think', 'going', 'go', 'really', 'say', 'come', \
        'hey', 'got', 'na', 'ca', 'look', 'good', 'oh', 'like', 'would']
stoplist.extend(more)

# get top n most frequent words in list of normalized tokens
# takes in list of list of tokens, returns list of top words
def getTopWords(lines, n):
    words = list(chain.from_iterable(lines)) # unnest
    words = [x for x in words if x not in stoplist] # extract stopwords
    fdist = nltk.FreqDist(words)
    tops = fdist.most_common(n)
    return [x[0] for x in tops]

# get top n words for a character that aren't in top 100 words of the show overall
def getTopWordsUncommon(df, character, n):
    overall_tops = getTopWords(list(df['Line']), 100) # top 100 words in the show
    char_lines = getLines(df, character) # lines for Sheldon
    words = list(chain.from_iterable(char_lines)) # unnest
    words = [x for x in words if x not in stoplist] # extract stopwords
    words = [x for x in words if x not in overall_tops] # extract super common words
    fdist = nltk.FreqDist(words)
    tops = fdist.most_common(n)
    return [x[0] for x in tops]

# get list of top words for each character in a show
def getCharTopWords(show):
    tops_list = []
    df = pd.read_pickle(join(your_path, 'datasets/norm_text_data/'+show+'Train.pkl'))
    characters = getCast(df) # list of main characters
    for character in characters: # for each character
        tops = getTopWordsUncommon(df, character, 20) # get top 20 words for this character
        # print(character, '\n', tops) # testing
        tops_list.append(tops)
    return tops_list


# testing
# os.chdir(your_path+'/features')
# os.chdir(your_path)
# tops_list = getCharTopWords('bang')
# tops_list = getCharTopWords('simpsons')
# tops_list = getCharTopWords('desperate')
