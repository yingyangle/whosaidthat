#this file get top words in traing set 

from getData import getData
from getData import normalizeData
import os, re, nltk, math, pandas as pd
from nltk.corpus import stopwords
from itertools import chain
import urllib

#let's bring in the normalized_data into this file 
global normalized_data
global stoplist

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

def get_list(normalized_data):
    #make a copy of normalized_data, so we don't change it in other files 
    copy = normalized_data[:]
    #unnest normalized_data
    copy = list(chain.from_iterable(copy))
    #extraxt stopwords 
    new_copy = []
    for i in copy:
        if i not in stoplist:
            new_copy.append(i)
    #get frequency distribution and append words to list    
    fdist = nltk.FreqDist(new_copy)
    final_list = []
    for i in fdist.most_common(20):
        final_list.append(i[0])
    return final_list

get_list(normalized_data)

