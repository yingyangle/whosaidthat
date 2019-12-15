import os
from matplotlib import pyplot as plt
from os.path import join

# average feature values
sheldon = [19.43, 3.233, 0.894, 0.391, 0.293, 0.139, 0.009, 0.32, 0.077, 0.309, 0.043, 0.0, 0.113, 0.085, 0.089, 0.093, 0.086, 0.096, 0.179]
leonard = [13.319, 2.966, 0.923, 0.383, 0.325, 0.069, 0.017, 0.274, 0.069, 0.327, 0.03, 0.0, 0.086, 0.079, 0.088, 0.13, 0.099, 0.081, 0.074]
howard = [14.709, 3.004, 0.92, 0.387, 0.31, 0.092, 0.028, 0.296, 0.077, 0.32, 0.047, 0.0, 0.085, 0.116, 0.172, 0.125, 0.102, 0.099, 0.08]
raj = [15.879, 3.022, 0.91, 0.391, 0.306, 0.091, 0.029, 0.312, 0.09, 0.32, 0.044, 0.0, 0.11, 0.099, 0.113, 0.119, 0.11, 0.19, 0.099]
bernadette = [13.134, 2.956, 0.934, 0.403, 0.311, 0.069, 0.018, 0.297, 0.073, 0.329, 0.033, 0.0, 0.093, 0.196, 0.105, 0.094, 0.096, 0.087, 0.056]
amy = [14.354, 3.069, 0.924, 0.396, 0.311, 0.087, 0.013, 0.306, 0.09, 0.311, 0.016, 0.0, 0.172, 0.086, 0.08, 0.094, 0.093, 0.084, 0.102]
penny = [13.352, 2.865, 0.92, 0.364, 0.345, 0.058, 0.024, 0.304, 0.089, 0.368, 0.04, 0.0, 0.089, 0.085, 0.077, 0.111, 0.138, 0.082, 0.069]

bart = [11.327, 3.07, 0.931, 0.303, 0.362, 0.069, 0.02, 0.244, 0.049, 0.26, 0.352, 0.112, 0.177, 0.076, 0.119, 0.068, 0.048]
homer = [13.022, 3.114, 0.916, 0.314, 0.352, 0.086, 0.018, 0.276, 0.06, 0.273, 0.392, 0.176, 0.082, 0.158, 0.086, 0.102, 0.057]
marge = [12.373, 3.205, 0.936, 0.338, 0.337, 0.077, 0.007, 0.271, 0.068, 0.243, 0.327, 0.128, 0.053, 0.081, 0.097, 0.147, 0.054]
lisa = [12.249, 3.206, 0.934, 0.323, 0.353, 0.078, 0.005, 0.26, 0.056, 0.257, 0.383, 0.13, 0.08, 0.071, 0.136, 0.087, 0.041]
flanders = [14.855, 3.226, 0.913, 0.313, 0.347, 0.074, 0.016, 0.277, 0.068, 0.249, 0.396, 0.191, 0.072, 0.081, 0.072, 0.126, 0.311]

susan = [15.251, 3.005, 0.902, 0.364, 0.337, 0.065, 0.022, 0.303, 0.066, 0.395, 0.163, 0.0, 0.076, 0.092, 0.096, 0.199]
gabrielle = [15.956, 3.059, 0.907, 0.37, 0.326, 0.083, 0.042, 0.307, 0.055, 0.406, 0.208, 0.0, 0.083, 0.176, 0.089, 0.089]
bree = [15.628, 3.19, 0.916, 0.384, 0.314, 0.068, 0.014, 0.293, 0.054, 0.348, 0.097, 0.0, 0.233, 0.088, 0.079, 0.062]
lynette = [15.103, 3.029, 0.906, 0.347, 0.341, 0.086, 0.025, 0.294, 0.068, 0.385, 0.149, 0.0, 0.081, 0.095, 0.175, 0.089]

bangFeats = [sheldon, leonard, howard, raj, bernadette, amy, penny]
bangCharacters = ['Sheldon', 'Leonard', 'Howard', 'Raj', 'Bernadette', 'Amy', 'Penny']
simpsonsFeats = [bart, homer, marge, lisa, flanders]
simpsonsCharacters = ['Bart', 'Homer', 'Marge', 'Lisa', 'Flanders']
desperateFeats = [susan, gabrielle, bree, lynette]
desperateCharacters = ['Susan', 'Gabrielle', 'Bree', 'Lynette']

def plot(x, y, characters):
    # plt.scatter(utterLen, wordLen, c='c') # training accuracy
    fig, ax = plt.subplots()
    ax.scatter(x, y, c='c')
    for i, txt in enumerate(characters):
        ax.annotate(txt, (x[i], y[i]), bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.2)) 
    plt.xlabel('Utterance Length', fontsize=18)
    plt.ylabel('Word Length', fontsize=18)
    plt.title('Big Bang Theory', fontsize=18)
    plt.savefig('plot.png', dpi=200) # save graph as png image
    return

def plotAll(x, y, characters):
    fig, ax = plt.subplots()
    c = ['blue', 'orange', 'green']
    for i in range(len(x)):
        ax.scatter(x[i],y[i],color=c[i])
        for k, txt in enumerate(characters[i]):
            ax.annotate(txt, (x[i][k], y[i][k]), bbox=dict(boxstyle='round,pad=0.2', fc=c[i], alpha=0.2))
    plt.xlabel('Subjectivity', fontsize=18)
    plt.ylabel('Polarity', fontsize=18)
    # plt.title('All Characters', fontsize=18)
    plt.savefig('plot.png', dpi=200) # save graph as png image
    return



# all 3 shows
charFeats = [bangFeats, simpsonsFeats, desperateFeats]
charList = [bangCharacters, simpsonsCharacters, desperateCharacters]
subjectivity = [[x[7] for x in y] for y in charFeats]
polarity = [[x[8] for x in y] for y in charFeats]
utterLen = [[x[0] for x in y] for y in charFeats]
wordLen = [[x[1] for x in y] for y in charFeats]
neologisms = [[x[4] for x in y] for y in charFeats]
profanity = [[x[6] for x in y] for y in charFeats]
plotAll(subjectivity, polarity, charList)


# just 1 show
charFeats = charFeats[0]
charList = charList[0]
subjectivity = [x[7] for x in charFeats]
polarity = [x[8] for x in charFeats]
utterLen = [x[0] for x in charFeats]
wordLen = [x[1] for x in charFeats]
neologisms = [x[4] for x in charFeats]
profanity = [x[6] for x in charFeats]
# plot(utterLen, wordLen, charList)
