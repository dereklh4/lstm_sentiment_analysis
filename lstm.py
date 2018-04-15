import numpy as np

#load pre-trained word embeddings
word_list = np.load("wordsList.npy").tolist()
word_list = [word.decode("UTF-8") for word in word_list]
word_vectors = np.load("wordVectors.npy")

assert len(word_list) == 400000
assert wordVectors.shape[1] = 50

#example using it
badger_index = word_list.index('badger')
print(word_vectors[badger_index])