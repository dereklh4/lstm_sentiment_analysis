import numpy as np
import os
from random import shuffle

dir_path = "review_polarity/txt_sentoken/"
pos_path = dir_path + "pos/"
neg_path = dir_path + "neg/"

np.random.seed(23) #DONT CHANGE THIS NUMBER. That would change the test set we are extracting
test_size = .3

#read in file paths
positive_review_files = [pos_path + f for f in os.listdir(pos_path) if os.path.isfile(os.path.join(pos_path, f))]
negative_review_files = [neg_path + f for f in os.listdir(neg_path) if os.path.isfile(os.path.join(neg_path, f))]

#split
test_positive_review_files = np.random.choice(positive_review_files,size=int(len(positive_review_files)*test_size),replace=False).tolist()
train_positive_review_files = [x for x in positive_review_files if x not in test_positive_review_files]

test_negative_review_files = np.random.choice(negative_review_files,size=int(len(negative_review_files)*test_size),replace=False).tolist()
train_negative_review_files = [x for x in negative_review_files if x not in test_negative_review_files]

train_review_files = train_positive_review_files + train_negative_review_files
shuffle(train_review_files)
test_review_files = test_positive_review_files + test_negative_review_files
shuffle(test_review_files)

#save list of files in each split
train_file = open('review_polarity/train_file_paths.txt', 'w+')
for file in train_review_files:
  train_file.write("%s\n" % file)

test_file = open('review_polarity/test_file_paths.txt', 'w+')
for file in test_review_files:
  test_file.write("%s\n" % file)
