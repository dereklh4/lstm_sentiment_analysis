from os import listdir
from os.path import isfile, join
import re
import string

dir_path = "review_polarity/txt_sentoken/"
pos_path = dir_path + "pos/"
neg_path = dir_path + "neg/"
positive_reviews = [pos_path + f for f in listdir(pos_path) if isfile(join(pos_path, f))]
negative_reviews = [neg_path + f for f in listdir(neg_path) if isfile(join(neg_path, f))]
numWords = []
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

for pf in positive_reviews:
    with open(pf, "r") as f:
        line=f.read()
        line = re.sub(strip_special_chars, "", line.lower())
        counter = len(line.split())
        numWords.append(counter)       

for nf in negative_reviews:
    with open(nf, "r") as f:
        line=f.read()
        line = re.sub(strip_special_chars, "", line.lower())
        counter = len(line.split())
        numWords.append(counter)

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))


import matplotlib.pyplot as plt
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
#plt.axis([0, 200, 0, 8000])
plt.show()