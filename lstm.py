import numpy as np
import tensorflow as tf
import os
import sys
import re

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_review(review_string):
	review_string = review_string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", review_string.lower())

#hyperparameters
maxSeqLength = 1500

#load pre-trained word embeddings
word_list = np.load("embeddings/wordsList.npy").tolist()
word_list = [word.decode("UTF-8") for word in word_list]
word_vectors = np.load("embeddings/wordVectors.npy")

assert len(word_list) == 400000
assert word_vectors.shape[1] == 50

#example using it
#badger_index = word_list.index('badger')
#print(word_vectors[badger_index])

# Create (or load) id matrix. It has converts words in reviews to their ids in the embeddings
dir_path = "review_polarity/txt_sentoken/"
pos_path = dir_path + "pos/"
neg_path = dir_path + "neg/"
positive_review_files = [pos_path + f for f in os.listdir(pos_path) if os.path.isfile(os.path.join(pos_path, f))]
negative_review_files = [neg_path + f for f in os.listdir(neg_path) if os.path.isfile(os.path.join(neg_path, f))]
review_files = positive_review_files + negative_review_files

if os.path.isfile("ids_matrix.npy"): #load it if available
	id_matrix = np.load("ids_matrix.npy")
else: #make it
	id_matrix = np.zeros((len(review_files), maxSeqLength), dtype='int32')
	total = len(review_files)
	file_counter = 0
	for rf in review_files:
		
		#print progress
		sys.stdout.write('\r')
		sys.stdout.write(str(file_counter) + "/" + str(total))
		sys.stdout.flush()

		with open(rf,"r") as f:
			index_counter = 0
			review_string=f.read()
			review_string = clean_review(review_string)
			split = review_string.split()
			for word in split:
			   try:
				   id_matrix[file_counter][index_counter] = word_list.index(word)
			   except ValueError:
				   id_matrix[file_counter][index_counter] = 399999 #Vector for unkown words
			   index_counter = index_counter + 1
			   if index_counter >= maxSeqLength:
				   break
			file_counter = file_counter + 1

	np.save("ids_matrix.npy",id_matrix)


