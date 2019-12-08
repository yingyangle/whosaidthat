# Christine Yang
# NLP Project: whosaidthat
# getData.py
# get data from .csv, normalize text, split into testing and training data
# save full, testing, and training datasets as pickles

import os, re, nltk, math, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from os.path import join

#your_path = '/Users/Christine/Documents/cs/whosaidthat' # christine
# your_path = '/Users/user/NLP Project/whosaidthat' # dora
your_path = "/Users/julianafakhoury/Documents/BC/nlp_project/recent/whosaidthat" # juliana


# get lines from df for a character or list of characters
def getLines(df, characters):
    if type(characters) is str: # if only looking for one character
        lines = list(df.loc[df.Speaker == characters].iloc[:, 1:]['Line'])
    else: # if we want lines for a list of characters
        lines = list(df[df['Speaker'].isin(characters)].iloc[:, 1:]['Line'])
    return lines

# get list of main characters in show
def getCast(df):
    cast = list(df.Speaker.unique())
    cast.sort()
    return cast

# split data in with n percent for testing, rest for training
# returns 2 dfs for training and testing data
def splitData(filename, n):
    # df of all original text data
    df = pd.read_csv(filename) #, encoding='ISO-8859-1')
    x = df.iloc[:,1:2].values # lines
    y = df.iloc[:,0:1].values # speakers
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=n)
    train = pd.DataFrame({'Speaker': y_train.flatten(), 'Line': X_train.flatten()})
    test = pd.DataFrame({'Speaker': y_test.flatten(), 'Line': X_test.flatten()})
    return (train, test)

# normalize list of str lines
def normalizeLines(lines):
    lem = WordNetLemmatizer()
    # original is a nested list of str: [[str],[str],...]
    # list to store a list of tokens for each utterance. [['token','token',...],['token','token',...]]
    tokenizedLines = []
    # tokenization
    for line in lines:
        utterance = str(line)
        utterance = utterance.lower() # lower case everything
        utterance = text2int(utterance) # text to number: twenty-six= 26
        tokens = nltk.word_tokenize(utterance) # tokenize
        lemmas = [lem.lemmatize(word) for word in tokens] # lemmatize
        tokenizedLines.append(lemmas)
    return tokenizedLines

# normalize lines in df
def normalizeDF(df):
    lines = np.array(df['Line']) # list of lines for all characters
    speakers = np.array(df['Speaker']) # list of speakers for each line
    normLines = normalizeLines(lines) # normalize lines
    data = {'Speaker': speakers, 'Line': normLines}
    new_df = pd.DataFrame(data) # dataframe of features for each line and 1/0 label
    return new_df

# convert all number words in utterance to digit numbers
def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        tens = [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"
        ]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

       # numwords["and"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10**(idx * 3 or 2), 0)

    ordinal_words = {
        'first': 1,
        'second': 2,
        'third': 3,
        'fifth': 5,
        'eighth': 8,
        'ninth': 9,
        'twelfth': 12
    }
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring

# get full, training, and testing datasets as original text and normalized tokens
# save each as pickle
def go():
    shows = ['bang', 'simpsons', 'desperate']
    for show in shows:
        # save full text dataset as pickle
        df = pd.read_csv(show+'.csv')
        df.to_pickle(join(your_path,'datasets/text_data/'+show+'Full.pkl'))
        # save train/test text datasets as pickles
        train, test = splitData(show+'.csv', 0.2)
        train.to_pickle(join(your_path,'datasets/text_data/'+show+'Train.pkl'))
        test.to_pickle(join(your_path,'datasets/text_data/'+show+'Test.pkl'))
        # save full normalized dataset as pickle
        norm_df = normalizeDF(df)
        norm_df.to_pickle(join(your_path,'datasets/norm_text_data/'+show+'Full.pkl'))
        # save train/test normalized datasets as pickles
        norm_train = normalizeDF(train)
        norm_train.to_pickle(join(your_path,'datasets/norm_text_data/'+show+'Train.pkl'))
        norm_test = normalizeDF(test)
        norm_test.to_pickle(join(your_path,'datasets/norm_text_data/'+show+'Test.pkl'))
        print('Finished', show)


# execute
# os.chdir(your_path)
# go()
