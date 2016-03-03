# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os

import GaussianGenerativeModel

from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

from sklearn.linear_model import LogisticRegression

from sklearn.gaussian_process import GaussianProcess

from sklearn import svm

from sklearn import tree

import util

TRAIN_DIR = "train"

call_set = set([])

def add_to_set(tree):

    for el in tree.iter():
        call = el.tag
        call_set.add(call)
        

def countAllSystemCalls(start_index, end_index, direc="train"):
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
 
        
def create_data_matrix(start_index, end_index, direc="train"):
    countAllSystemCalls(start_index, end_index, direc="train")
    print "len ", len(call_set)
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        #add_to_set(tree)
        #print "length of call set is ", len(call_set)
        this_row = call_feats(tree)
        
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids
    

def call_feats(tree):
    #good_calls = ['sleep', 'dump_line']
    good_calls = list(call_set)
    #print "good calls is ", len(good_calls)

        
    
    call_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1
    
    #print "length of call_counter is ", len(call_counter)
    

    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]

    return call_feat_array

## Feature extraction
def main():
    #X_train, t_train, train_ids = create_data_matrix(0, 3068, TRAIN_DIR)
    #X_valid, t_valid, valid_ids = create_data_matrix(0, 3724, "test")
    
    X_train, t_train, train_ids = create_data_matrix(0, 3086, TRAIN_DIR)
    X_valid, t_valid, valid_ids = create_data_matrix(0, 3724, "test")
    #print "call set is ", X_train
    
    '''model = LogisticRegression()
    
    model.fit(X_train, t_train)
    
    probs = model.predict(X_valid)'''
    
    model = svm.SVC()
    model = tree.DecisionTreeClassifier()
    
    model.fit(X_train, t_train)
    
    probs = model.predict(X_valid)
    
    #print "probs are ", probs
    
    #util.write_predictions(probs, valid_ids, "junk.csv")
    

    '''print 'Data matrix (training set):'
    print "len of X is ", len(X_train[0])
    print 'Classes (training set):'
    print t_train'''

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()
    