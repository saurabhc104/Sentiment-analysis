#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:47:04 2017

@author: saurabh
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import re
import math
import tensorflow as tf
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model



def upload_glove():
    gloVe = open('glove.6B/glove.6B.50d.txt','r')
    wordsList = []
    wordVectors = []
    for line in gloVe:
        splitted_text = line.split()
        wordsList.append(splitted_text[0])
        wordVectors.append(splitted_text[1:])
        
    wordVectors = np.array(wordVectors)
    print('Number of words in gloVe model: ', len(wordVectors))
    
    return wordVectors, wordsList


def get_file_analysis():
    positiveFiles = ['pos/' + f for f in listdir('pos/') if isfile(join('pos/', f))]
    negativeFiles = ['neg/' + f for f in listdir('neg/') if isfile(join('neg/', f))]
    numWords = []
    
    
    for pf in positiveFiles:
        with open(pf, "r", encoding='utf-8') as f:
            line=f.readline()
            counter = len(line.split())
            numWords.append(counter) 
            
    for nf in negativeFiles:
        with open(nf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            numWords.append(counter)
            
    print("Total number of reviews: ",len(numWords))
    print("Total number of words in whole file: ",sum(numWords))
    print("Average length of review: ", sum(numWords)/len(numWords))
    
    return 50*math.ceil((sum(numWords)/len(numWords))/50), numWords, positiveFiles, negativeFiles




def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def cerateTensor(numWords, maxSeqLength, positiveFiles, negativeFiles):
    
    numReviews = len(numWords)
    
    try:
        ids = np.load('idsMatrix.npy')
        output = np.load('output.npy')
    except:
        ids = np.zeros((numReviews, maxSeqLength))
        output = np.zeros((numReviews,2))  #positive = [1,0], negative = [0,1]
        fileCounter = 0
        for pf in positiveFiles:
            with open(pf, "r") as f:
                output[fileCounter] = [1, 0]
                indexCounter = 0
                line=f.readline()
                cleanedLine = cleanSentences(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        ids[fileCounter][indexCounter] = wordsList.index(word)
                    except ValueError:
                        ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                    indexCounter = indexCounter + 1
                    if indexCounter >= maxSeqLength:
                        break
                fileCounter = fileCounter + 1 
        
        for nf in negativeFiles:
            with open(nf, "r") as f:
                output[fileCounter] = [0, 1]
                indexCounter = 0
                line=f.readline()
                cleanedLine = cleanSentences(line)
                split = cleanedLine.split()
                for word in split:
                    try:
                        ids[fileCounter][indexCounter] = wordsList.index(word)
                    except ValueError:
                        ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                    indexCounter = indexCounter + 1
                    if indexCounter >= maxSeqLength:
                        break
                fileCounter = fileCounter + 1
        np.save('idsMatrix', ids)
        np.save('output', output)
    return ids, output



#create_embedding_lookup
#Build final tensor of shape 25000,250,50
def create_embedding_lookup(ids):
    try:
        feedable_tensor = np.load("feeable_tensor.npy")
    except:
        feedable_tensor = np.zeros((25000, 250, 50))
        
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                feedable_tensor[i][j] = wordVectors[ids[i][j]]
                
            
        np.save("feedable_tensor", feedable_tensor)
    return feedable_tensor


def get_train_test(feedable_tensor, output, positiveFiles, negativeFiles):
    #Train set
    X_train = feedable_tensor[:int(len(positiveFiles)*.9)]
    X_train = np.append(X_train, feedable_tensor[len(positiveFiles):len(positiveFiles)+int(len(negativeFiles)*.9)], axis=0)
    
    Y_train = output[:int(len(positiveFiles)*.9)]
    Y_train = np.append(Y_train, output[len(positiveFiles):len(positiveFiles)+int(len(negativeFiles)*.9)], axis=0)
    
    
    #Test set
    X_test = feedable_tensor[int(len(positiveFiles)*.9):len(positiveFiles)]
    X_test = np.append(X_test, feedable_tensor[len(positiveFiles)+int(len(negativeFiles)*.9):], axis=0)
    
    Y_test = output[int(len(positiveFiles)*.9):len(positiveFiles)]
    Y_test = np.append(Y_test, output[len(positiveFiles)+int(len(negativeFiles)*.9):], axis=0)
    return X_train, Y_train, X_test, Y_test


def buildModel(x_train, y_train, x_test, y_test):
    try:
        model = load_model('pretrained.h5')
    except:
        model = Sequential()
        model.add(LSTM(128, input_shape=(250,50)))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train, y_train,  epochs=20, batch_size=64, verbose=1, validation_data=(x_test, y_test))
        model.save('pretrained.h5') 
    output = model.predict(x_test)
    return output
 

    
if __name__ == "__main__":
    wordVectors, wordsList = upload_glove()
    print("upload_glove done")
    maxSeqLength, numWords, positiveFiles, negativeFiles = get_file_analysis()
    print("get_file_analysis done")
    
    ids, output = cerateTensor(numWords, maxSeqLength, positiveFiles, negativeFiles)
    ids = ids.astype('int64')
    print("createTensor done")
    
    feedable_tensor = create_embedding_lookup(ids)
    print("create_embedding_lookup done")
    
    X_train, Y_train, X_test, Y_test = get_train_test(feedable_tensor, output, positiveFiles, negativeFiles)
    print("get_train_get_test done")
    print("Building model...")
    test_output = buildModel(X_train, Y_train, X_test, Y_test)
    





#
#def getTrainBatch():
#    labels = []
#    arr = np.zeros([batchSize, maxSeqLength])
#    for i in range(batchSize):
#        if (i % 2 == 0): 
#            num = randint(1,11499)
#            labels.append([1,0])
#        else:
#            num = randint(13499,24999)
#            labels.append([0,1])
#        arr[i] = ids[num-1]
#    return arr, labels
#
#def getTestBatch():
#    labels = []
#    arr = np.zeros([batchSize, maxSeqLength])
#    for i in range(batchSize):
#        num = randint(11499,13499)
#        if (num <= 12499):
#            labels.append([1,0])
#        else:
#            labels.append([0,1])
#        arr[i] = ids[num-1]
#    return arr, labels


#-------LSTM-------
#
#batchSize = 24
#lstmUnits = 64
#numClasses = 2
#iterations = 100000
#
#tf.reset_default_graph()
#labels = tf.placeholder(tf.float32, [batchSize, numClasses])
#input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
#
#data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
#data = tf.nn.embedding_lookup(wordVectors,input_data)
#
#lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
#lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
#value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#
#
#weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
#bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
#value = tf.transpose(value, [1, 0, 2])
#last = tf.gather(value, int(value.get_shape()[0]) - 1)
#prediction = (tf.matmul(last, weight) + bias)
#
#correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
#accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
#optimizer = tf.train.AdamOptimizer().minimize(loss)
#
#import datetime
#
#sess = tf.InteractiveSession()
#tf.summary.scalar('Loss', loss)
#tf.summary.scalar('Accuracy', accuracy)
#merged = tf.summary.merge_all()
#logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
#writer = tf.summary.FileWriter(logdir, sess.graph)
#
#
#
#saver = tf.train.Saver()
#sess.run(tf.global_variables_initializer())
#
#for i in range(iterations):
#    
#    #Next Batch of reviews
#    nextBatch, nextBatchLabels = getTrainBatch();
#    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#    
#    #Write summary to Tensorboard
#    if (i % 50 == 0):
#        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#        writer.add_summary(summary, i)
#
#    #Save the network every 10,000 training iterations
#    if (i % 10000 == 0 and i != 0):
#        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
#        print("saved to %s" % save_path)
#writer.close()
#
#
#sess = tf.InteractiveSession()
#saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('models'))
#
#iterations = 10
#for i in range(iterations):
#    nextBatch, nextBatchLabels = getTestBatch();
#    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
#
#
# 