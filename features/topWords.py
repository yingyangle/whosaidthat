# topWords.py
# get the top N most frequent words in a dataset of normalized text data

import os, re, nltk
from nltk.corpus import stopwords
from itertools import chain
from getData import getData, normalizeData

# list of stopwords to exclude when getting most frequent words list
stoplist = stopwords.words('english')
stoplist.append(",")
stoplist.append(".")
stoplist.append('“')
stoplist.append('”')
stoplist.append(';')
stoplist.append('?')
stoplist.append('--')
stoplist.append('’')
stoplist.append("n't")
stoplist.append('would')
stoplist.append('!')
stoplist.append('could')
stoplist.append("'s")

# get top n most frequent words in normalized data
# takes in list of list of tokens, returns list
def getTopWords(lines, n):
    # make a copy of normalized_data, so we don't change it in other files
    copy = lines[:]
    # unnest normalized_data
    copy = list(chain.from_iterable(copy))
    # extraxt stopwords
    new_copy = []
    for i in copy:
        if i not in stoplist:
            new_copy.append(i)
    # get frequency distribution and append words to list
    fdist = nltk.FreqDist(new_copy)
    final_list = []
    for i in fdist.most_common(n):
        final_list.append(i[0])
    return final_list

# testing
# data = getData('bang.csv', 'Sheldon')
# data = normalizeData(data)
# print(getTopWords(data, 20))
