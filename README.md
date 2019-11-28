# whosaidthat

## Features
| 0  | utterance length        | number of words in the utterance                                                              |
|----|-------------------------|-----------------------------------------------------------------------------------------------|
| 1  | average word length     | average length of words in the utterance                                                      |
| 2  | utterance type          | [statement, question, exclamation, etc.]                                                      |
| 3  | top words               | number of words in this utterance that are also in the character's top 20 most frequent words |
| 4  | sentiment               | [positive, negative, neutral, etc.]                                                           |
| 5  | diversity of vocabulary | type-token ratio for this utterance                                                           |
| 6  | number of neologisms    | number of words in this utterance that are not in our vocabulary                              |
| 7  | stop words              | percentage of words in this utterance that are stop words                                     |
| 8  | POS ratio               |                                                                                               |
| 9  | contains numbers        | True/False: does this utterance contain numbers?                                              |
| 10 | contains profanity      | True/False: does this utterance contain words from our profanity list?                        |
