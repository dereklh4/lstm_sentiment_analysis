from collections import Counter
import string
import os
import tensorflow as tf
import numpy as np

#get stopwords
stopwords = set()
with open("bag_data/stopwords_english.txt") as f:
	words = f.read().splitlines()
	for word in words:
		stopwords.add(word)

def generate_tokens(review):
	# split into tokens by white space
	tokens = review.split()
	# remove punctuation from each token
	#table = string.maketrans(string.punctuation,''*len(string.punctuation))
	tokens = [w.translate(None,string.punctuation) for w in tokens]
	#s.translate(None, string.punctuation)
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	tokens = [w for w in tokens if not w in stopwords]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

def add_doc_to_vocab(filename, vocab):
	# load doc
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	# clean doc
	tokens = generate_tokens(doc)
	# update counts
	vocab.update(tokens)

#read in stopwords
def get_vocabulary():
	file_path = "bag_data/vocab.txt"
	if os.path.isfile(file_path):
		with open(file_path) as f:
			tokens = f.read().splitlines()
	else:

		#build vocabulary
		vocab = Counter()
		with open("review_polarity/train_file_paths.txt") as f:
			train_review_files = f.read().splitlines()

		for file in train_review_files:
			add_doc_to_vocab(file,vocab)

		# keep tokens with a min occurrence
		min_occurane = 2
		tokens = [k for k,c in vocab.items() if c >= min_occurane]

		#save for later
		f = open(file_path,"w+")
		data = "\n".join(tokens)
		f.write(data)
		f.close()
	
	return tokens

# load docs into a list. Each element in list is all the words of a single document
def load_docs(filenames,vocab):
	docs = list()
	# walk through all files in the folder
	for filename in filenames:
		#read it
		file = open(filename,"r")
		text = file.read()
		file.close()
		#generate tokens
		doc_tokens = generate_tokens(text)
		# filter by vocab
		doc_tokens = [w for w in doc_tokens if w in vocab]
		#get as string
		review_words = ' '.join(doc_tokens)
		# add to list
		docs.append(review_words)
	return docs

def get_encoded_data(vocab,train=True,mode="freq"):
	
	with open("review_polarity/train_file_paths.txt") as f:
		train_review_files = f.read().splitlines()
	with open("review_polarity/test_file_paths.txt") as f:
		test_review_files = f.read().splitlines()

	train_data_filepath = "bag_data/" + "train" + "_" + mode + ".npy"
	test_data_filepath = "bag_data/" + "test" + "_" + mode + ".npy"
	train_savepath = "bag_data/" + "train" + "_" + mode
	test_savepath = "bag_data/" + "test" + "_" + mode
	labels = []
	test_labels = []
	if os.path.isfile(train_data_filepath):
		print("Loading saved encoded dataset")
		X = np.load(train_data_filepath)
		Xtest = np.load(test_data_filepath)
	else:
		#use tokenizer to encode a review as a bag of words
		tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer()
		print("Loading training docs")
		train_docs = load_docs(train_review_files,vocab)
		print("Loading test docs")
		test_docs = load_docs(test_review_files,vocab)
		print("Fitting tokenizer")
		tokenizer.fit_on_texts(train_docs)

		#encode training set
		X = tokenizer.texts_to_matrix(train_docs, mode=mode)
		Xtest = tokenizer.texts_to_matrix(test_docs,mode=mode)

		np.save(train_savepath,X)
		np.save(test_savepath,Xtest)

	#get labels
	for file in train_review_files:
		label = 1 if string.find(file,"pos") >= 0 else 0
		labels.append(label)
	for file in test_review_files:
		label = 1 if string.find(file,"pos") >= 0 else 0
		test_labels.append(label)
		
	return X,labels,Xtest,test_labels

#get vocab for training set
with open("review_polarity/train_file_paths.txt") as f:
	train_review_files = f.read().splitlines()
vocab = get_vocabulary()

#get encoded dataset
Xtrain,train_labels,Xtest,test_labels = get_encoded_data(vocab,True,"binary")
print(Xtrain.shape)
print(len(train_labels))
print(Xtest.shape)
print(len(test_labels))

#try a classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression

#clf = RandomForestClassifier()
#clf = tree.DecisionTreeClassifier()
#clf = svm.SVC()
clf = LogisticRegression(C = 100.0, random_state = 1)

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
clf = LogisticRegression(C = 100.0, random_state = 1)
print("Training classifier")
clf.fit(Xtrain,train_labels)

#get training stats
train_predictions = clf.predict(Xtrain)
precision,recall,fscore,support = precision_recall_fscore_support(train_labels,train_predictions, average='macro')
print("--Training Set--")
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("FScore: " + str(fscore))
accuracy = len(np.where(train_labels == train_predictions)[0]) / (1.0 * len(train_labels))
print("Accuracy: " + str(accuracy))

#get test stats
test_predictions = clf.predict(Xtest)
precision,recall,fscore,support = precision_recall_fscore_support(test_labels,test_predictions, average='macro')
print("--Test Set--")
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("FScore: " + str(fscore))
accuracy = len(np.where(test_labels == test_predictions)[0]) / (1.0 * len(test_labels))
print("Accuracy: " + str(accuracy))