# Christine Yang
# NLP Project: whosaidthat
# getFeatures.py
# extract features from data

# profanity word list based on:
# https://github.com/areebbeigh/profanityfilter/


# utterance length

# average word length

# utterance type

# top words

# sentiment

# diversity of vocabulary

# number of neologisms

# stop words

# POS ratio

# contains numbers

# contains profanity
ein = open('profanity.txt', 'r')
profanity = ein.read().rstrip().split('\n')
ein.close()
