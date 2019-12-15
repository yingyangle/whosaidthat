#!/Users/Christine/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Christine Yang
# NLP Project: whosaidthat
# random.py
# some random testing things/ experiments
# try getting average value of each feature for each character
# try making even dataset with same # lines per character

import os, re, nltk, math, pandas as pd, numpy as np, warnings, random, pickle
from statistics import mean
from os.path import join
from getData import getLines
warnings.filterwarnings("ignore")

your_path = '/Users/Christine/Documents/cs/whosaidthat' 


# execute
os.chdir(your_path)

shows = ['bang', 'simpsons', 'desperate']
bangDict = {'Amy': 0, 'Bernadette': 1, 'Howard': 2, 'Leonard': 3, 'Penny': 4, 'Raj': 5, 'Sheldon': 6}
simpsDict = {'Bart': 0, 'Homer': 1, 'Lisa': 2, 'Marge': 3, 'Ned Flanders': 4}
despDict = {'Bree':0, 'Gabrielle':1, 'Lynette':2, 'Susan':3}
dictDict = {'bang':bangDict, 'simpsons':simpsDict, 'desperate':despDict}


# df = pd.read_pickle(join(your_path, 'datasets/norm_text_data/bangTest.pkl'))
# print(df)


# count lines for each character -- general dataset info

# for show in shows: # for each show
#     df = pd.read_pickle(join(your_path, 'datasets/text_data/'+show+'Full.pkl'))
#     characters = list(df.Speaker.unique())
#     print(show, "\nTotal lines: ", df.shape)
#     for character in characters:
#         char_lines = list(df.loc[df.Speaker == character].iloc[:, 1:]['Line'])
#         print(character, len(char_lines))
    

# get average

# for show in shows: # for each show
#     df = pd.read_pickle(join(your_path, 'datasets/features_data/'+show+'Train.pkl'))
#     characters = list(df.Label.unique())
#     print(characters)
#     charDict = dictDict[show]
#     print(charDict)
#     for character in characters:
#         print(character, charDict[character])
#         char_feats = list(df.loc[df.Label == character].iloc[:, 1:]['Features'])
#         list_feats = [[x[i] for x in char_feats] for i in range(len(char_feats[0]))]
#         avg_feats = [round(sum(ls) / len(ls), 3) for ls in list_feats]
#         print(avg_feats)


# get median

# for show in shows: # for each show
#     df = pd.read_pickle(join(your_path, 'datasets/features_data/'+show+'Train.pkl'))
#     characters = list(df.Label.unique())
#     print(characters)
#     charDict = dictDict[show]
#     print(charDict)
#     for character in characters:
#         print(character, charDict[character])
#         char_feats = list(df.loc[df.Label == character].iloc[:, 1:]['Features'])
#         list_feats = [[x[i] for x in char_feats] for i in range(len(char_feats[0]))]
#         mean_feats = [round(mean(ls), 3) for ls in list_feats]
#         print(mean_feats)


# show = 'bang'
# df = pd.read_pickle(join(your_path, 'datasets/features_data/'+show+'Train.pkl'))
# print(df['Features'][0])
# print(df['Features'][0])




# get same number of lines for each character

# for show in shows: # for each show
#     df = pd.read_pickle(join(your_path, 'datasets/features_data/'+show+'Train.pkl'))
#     characters = list(df.Label.unique())
#     min_num = float('inf')
#     min_char = -1
#     for character in characters:
#         char_lines = list(df.loc[df.Label == character].iloc[:, 1:]['Features'])
#         num_lines = len(char_lines)
#         if num_lines < min_num:
#             min_num = num_lines
#             min_char = character
#     print(show, min_num, min_char)
#     new_char_lines = []
#     new_char_labels = []
#     for character in characters:
#         if character == min_char: continue
#         char_lines = list(df.loc[df.Label == character].iloc[:, 1:]['Features'])
#         new_char_lines.extend(random.sample(char_lines, min_num))
#         new_char_labels.extend([character for x in range(min_num)])
#     data = {'Label': new_char_labels, 'Features': new_char_lines}
#     new_df = pd.DataFrame(data)
#     new_df.to_pickle('datasets/even_features_data/' + show + 'Train.pkl')
#
# show = 'bang'
# df = pd.read_pickle(join(your_path, 'datasets/even_features_data/'+show+'Train.pkl'))
# print(df)
#
        
    

