# Christine Yang
# NLP Project: whosaidthat
# bang.py
# get transcripts for The Big Bang Theory, save as .txt and .csv

import os, codecs, re, requests, pandas as pd
from bs4 import BeautifulSoup

os.chdir('/Users/Christine/cs/whosaidthat/raw_data')

# BIG BANG THEORY ####################################################################

fulltext = '' # one full transcript containing all episodes
links = [] # list of links to each episode transcript

sesh = requests.Session()
sesh.header = { 'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0","Accept-Encoding": "gzip, deflate, sdch"}

# get links to each episode transcript
def getLinks():
    links = []
    try: # see if links are already saved in .txt file
        ein = codecs.open('bang_links.txt', 'r', 'utf-8')
        raw = ein.read().rstrip()
        ein.close()
        links = raw.split('\n') # get links if already saved in .txt file
    except: # otherwise, collect links and save them in .txt file
        url = 'https://bigbangtrans.wordpress.com/' # homepage with links to each episode
        req = sesh.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        alinks = soup.findAll(id='pages-2')[0].findAll('a')
        for i in range(1, len(alinks)): # start from 1 since [0] is the about page
            links.append(alinks[i].get('href'))
        aus = codecs.open('bang_links.txt', 'w', 'utf-8')
        temp = [aus.write(l+'\n') for l in links]
        aus.close()
    return links

# save one episode transcript to fulltext
def getEpisode(url):
    global fulltext
    req = sesh.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    raw_text = soup.findAll('p')
    text = ''
    for i in range(len(raw_text)):
        line = raw_text[i].getText() # line of text
        if ':' not in line: continue
        text = text + line + '\n'
        text.count(':')
    fulltext = fulltext + text + '\n'
    print(text.count(':'), url) # print progress
    return text

# save all episode transcripts to one big .txt file
def getTranscript():
    global fulltext, links
    links = getLinks() # get links to each episode transcript
    for url in links:
        getEpisode(url) # save each episode transcript to fulltext
    fulltext = re.sub('\(.*\)', '', fulltext) # get rid of stuff in parentheses
    fulltext = re.sub(':\W+:', ':', fulltext) # get rid of accidental double colons
    fulltext = fulltext.replace(u'\xa0', u' ') # get rid of the weird \xa0 char
    aus = codecs.open('bang.txt', 'w', 'utf-8')
    aus.write(fulltext) # write fulltext to .txt file
    aus.close()

# format .txt transcript as .csv with speaker and line split up
def getCSV():
    ein = codecs.open('bang.txt', 'r', 'utf-8')
    raw = ein.read().rstrip() # get transcript from .txt
    ein.close()
    lines = raw.split('\n') # list of lines in transcript
    # split each line into speaker and line
    data = [x for x in [l.split(': ', 1) for l in lines] if len(x) is 2]
    df = pd.DataFrame(data, columns = ['Speaker', 'Line'])
    df.to_csv("bang.csv", index=False, encoding='utf-8-sig') # save to .csv

# execute
# getTranscript()
getCSV()
