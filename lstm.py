import numpy as np
import tensorflow as tf
import os
import sys
import re
import datetime
import random

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_review(review_string):
	review_string = review_string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", review_string.lower())

def getBatch(batchSize,num_reviews,id_matrix):
	"""Note: This assumes that the classes are evenly split (half positive, half negative, in that order)"""
	labels = []
	arr = np.zeros([batchSize, maxSeqLength])
	num_pos = num_reviews / 2
	num_neg = num_reviews - num_pos
	for i in range(batchSize):
		if (i % 2 == 0): #get pos
			num = random.randint(1,num_pos)
			labels.append([1,0])
		else: #get neg
			num = random.randint(num_pos,num_pos+num_neg)
			labels.append([0,1])
		arr[i] = id_matrix[num-1:num]
	return arr, labels

#hyperparameters
lstmUnits = 64
learning_rate = .001
iterations = 100000
dropout_keep_prob = .75
batchSize = 24
maxSeqLength = 1500
numDimensions = 50 #number of dimensions in a vector for a word

#other settings
run_train = False
run_test = True
numClasses = 2
dir_path = "review_polarity/txt_sentoken/"
pos_path = dir_path + "pos/"
neg_path = dir_path + "neg/"
train_ids_matrix_file = "train_ids_matrix_" + str(maxSeqLength) + ".npy"
test_ids_matrix_file = "test_ids_matrix_" + str(maxSeqLength) + ".npy"

#load pre-trained word embeddings
word_list = np.load("embeddings/wordsList.npy").tolist()
word_list = [word.decode("UTF-8") for word in word_list]
word_vectors = np.load("embeddings/wordVectors.npy")

assert len(word_list) == 400000
assert word_vectors.shape[1] == numDimensions

#example using it
#badger_index = word_list.index('badger')
#print(word_vectors[badger_index])

#get data
with open("review_polarity/train_file_paths.txt") as f:
	train_review_files = f.read().splitlines()
with open("review_polarity/test_file_paths.txt") as f:
	test_review_files = f.read().splitlines()

print("Num train review files:" + str(len(train_review_files)))
print("Num test review files:" + str(len(test_review_files)))

# Create (or load) id matrix for train and test set. It has converts words in reviews to their ids in the embeddings
for mode,review_files,ids_matrix_file in [("train",train_review_files,train_ids_matrix_file),("test",test_review_files,test_ids_matrix_file)]:
	if os.path.isfile(ids_matrix_file): #load it if available
		id_matrix = np.load(ids_matrix_file)
	else: #make it
		print("")
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

		np.save(ids_matrix_file,id_matrix)

	if mode == "test":
		test_ids_matrix = id_matrix
	else:
		train_ids_matrix = id_matrix

## MODEL ##

tf.reset_default_graph()

#tensorflow placeholders for inputs and labels
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

#get review word embedding data
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors,input_data)

#model
#TODO: Could make more sophisticated with stacked LSTMs (will take longer to train though)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob) #to reduce overfitting
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

#define accuracy, loss, and optimizer
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

## TRAIN ##

if run_train:
	#save summaries to track training progress on tensorboard. Use "tensorboard --logdir=tensorboard" to see it
	tf.summary.scalar('Loss', loss)
	tf.summary.scalar('Accuracy', accuracy)
	merged = tf.summary.merge_all()
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	writer = tf.summary.FileWriter(logdir, sess.graph)

	for i in range(iterations):
		
		#print progress
		sys.stdout.write('\r')
		sys.stdout.write(str(i) + "/" + str(iterations))
		sys.stdout.flush()

		#Next Batch of reviews
		nextBatch, nextBatchLabels = getBatch(batchSize,len(train_review_files),train_ids_matrix);
		sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

		#Write summary to Tensorboard
		if (i % 50 == 0):
		   summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
		   writer.add_summary(summary, i)

		#Save the network every 2,000 training iterations
		if (i % 2000 == 0 and i != 0):
			save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
			print("saved to %s" % save_path)
	writer.close()

## TEST ##

if run_test:
	iterations = 10
	for i in range(iterations):
		nextBatch, nextBatchLabels = getBatch(batchSize,len(test_review_files),test_ids_matrix)
		print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

