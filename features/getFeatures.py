# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/

from getData import getData
from nltk.corpus import words
from nltk.corpus import stopwords

allwords = words.words()
stopwords = stopwords.words('english')

# take in a line as a list of tokens

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
    return len(neologisms) / len(line)

# ratio of stop words to words in utterance
def stopwords(line):
    stops = [x for x in line if x in stopwords]
    return len(stops) / len(line)

# POS ratio


# contains numbers
# 1 = contains numbers, 0 = doesn't contain numbers
def hasNumbers(line):
    for num in range(10):
        if num in line: return 1
    return 0

# contains profanity
# 1 = contains profanity, 0 = doesn't contain profanity
def hasProfanity(line):
    ein = open('profanity.txt', 'r')
    profanity = ein.read().rstrip().split('\n')
    ein.close()
    for p in profanity:
        if p in line: return 1
    return 0
