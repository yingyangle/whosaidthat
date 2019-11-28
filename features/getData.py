# Christine Yang
# NLP Project: whosaidthat
# getData.py
# get dialogue lines from csv files and normalize 

import os, re, pandas as pd

# os.chdir('/Users/Christine/Documents/cs/whosaidthat')
os.chdir('/Users/user/NLP Project/whosaidthat/BangNewestData')


# get lines from filename for a character or list of characters
def getData(filename, characters):
    df = pd.read_csv(filename) # df = dataframe
    if type(characters) is str: # if only looking for one character
        lines = df.loc[df.Speaker == characters].iloc[:,1:].values
    else: # if we want lines for a list of characters
        lines = df[df['Speaker'].isin(characters)].iloc[:,1:].values
    print(lines)
    return lines

# normalize data
def normalizeData():
    return

# testing
# getData('bang.csv', 'Sheldon')
# getData('bang.csv', ['Sheldon', 'Leonard'])

# Dora testing
# getData('bangSevenChars.csv', 'Sheldon')
getData('bangSevenChars.csv', ['Sheldon', 'Leonard'])