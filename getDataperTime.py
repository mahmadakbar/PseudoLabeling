# -*- coding: utf-8 -*-

import zmq
import os

# connect and open to ryuData App
# os.system("xterm -e \"python2 ryu-restapi.py\" &")
# os.system("xterm -e \"python2 cpusingle.py\" &")
# time.sleep(5)

# print "Done"
print "Please Wait ....."

import numpy as np
import pandas as pd

import sched
import time
import subprocess
import sys
from twisted.internet import task, reactor

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import LabelEncoder
from subprocess import Popen, PIPE, STDOUT

import matplotlib.pyplot as plt
import seaborn as sns
from twisted.internet import task, reactor

import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DataConversionWarning)

import getopt
from scapy.all import *
from scapy import all as scapy
from random import randrange
from scapy.utils import PcapWriter
import strings

import requests
import json

def getType(whoIP):
    """
    get type for calculate precision, recall, f1score
    """
    arrayType = []
    for j in whoIP:
        if j == "192.168.3.17" or j == "192.168.3.18" or j == "192.168.3.33":
            arrayType.append('NORMAL')
        else:
            arrayType.append('DDOS')
    return arrayType

timeout = 10.0 # Sixty seconds

def doWork():
    ts = time.time()

    one = (ts)*1000
    time1 = int(one)

    two = (ts-10)*1000
    time2 = int(two)

    # print "time now = %s , and time 5 second pass = %s" % (one, two)
    print "time 10 second pass = %s , and time now = %s" % (time2, time1)
    subprocess.call(["mongoexport",
                     "--db",
                     "mnemosyne",
                     "--collection",
                     "session",
                     "--type=csv",
                     "--fields=_id,protocol,hpfeed_id,timestamp,source_ip,destination_ip,identifier,honeypot",
                     "-q", """{"timestamp":{$gt:new Date(%s),$lt:new Date(%s)}, "protocol" : "ICMP"}""" % (time2, time1),
                     "--out",
                     "output/%s.csv" % time1])
    
    dataset = pd.read_csv('icmp/dataSample/test.csv') # read & prepare data
    getDataset = pd.read_csv('output/%s.csv' % time1)
    getIP = getDataset['source_ip']
    aType = getType(getIP)
    trueType = pd.DataFrame(data=aType,columns=['type'])

    dataset2 = pd.concat([getDataset, trueType], axis = 1, join_axes=[getDataset.index])

    columnz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
    columnzz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot','type']

    dropUnsuable = dataset.drop(dataset.columns[0], axis=1) # drop type

    if not aType :
        print "\n------------ We dont have data to be process ------------\n"
        subprocess.call(["rm", "output/%s.csv" % time1])
        # subprocess.call(["rm", "output/encode/%s_encode.csv" % time1])
    else:
        print "\n----------------- data ready to process -----------------"

        joinData = pd.concat([dropUnsuable, dataset2], ignore_index=True)
        # joinData.to_csv(r'dataSample/testes.csv')

        Xa = joinData.drop(columns=['type'])
        Ya = joinData['type'].values

        labelencoder = LabelEncoder() # prepare for labelEncoder
        Xb = Xa.apply(labelencoder.fit_transform) # apply label encoder on "Xa"
        Yb = labelencoder.fit_transform(Ya) # apply label encoder on "Ya"

        Xb.to_csv(r'output/encode/%s_encode.csv' % time1)

        sc_X = StandardScaler() # prepare for StandardScaler
        X = sc_X.fit_transform(Xb) # apply label encoder on "X"

        Xscaler = pd.DataFrame(X, columns=columnz)
        Xscaler.to_csv(r'output/scaler/%s_scaler.csv' % time1)

        #split Train
        jm = (dataset.index[-1])

        arrayTrain = Xb[:jm]
        getYtrain = Yb[:jm]
        trainScalern = X[:jm]

        #split Test
        jmt = (dataset.index[-1])+1

        arrayTest = Xb[jmt:].reset_index(drop=True)
        getYtest = Yb[jmt:]
        testScalern = X[jmt:]

        dropTest = pd.DataFrame(testScalern, columns=columnz) # call 'X test' array and make them to dataframe

        # start execute data with ML algoritm >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        start = time.time() #timestart

        lin_clf = svm.LinearSVC()
        lin_clf.fit(trainScalern, getYtrain) # training "x train" and "y train"

        pseudoY_test = lin_clf.predict(dropTest) # data that won to predict by row
                

        X = np.vstack((trainScalern, testScalern))
        Y = np.concatenate((getYtrain, pseudoY_test), axis=0)

        pseudo_model = svm.LinearSVC()
        pseudo_model.fit(X, Y) # try to predict with pseudo using LinierSVC

        clf = AdaBoostClassifier(n_estimators=10)
        scores = cross_val_score(clf, X, Y) # predict again with AdaBoost
        scores.mean()
        clf.fit(X, Y)

        AccuracY = clf.score(X, Y)
        print "Accuracy : ", AccuracY*100, "%"

        stop = time.time()
        timeF = stop - start
        print "--- %s seconds ---" % timeF
                
        prediction = clf.predict(testScalern)
        allScore = precision_recall_fscore_support(getYtest, prediction, average='micro')
        print "Precision    : ", allScore[0]
        print "Recall       : ", allScore[1]
        print "f1 Socre     : ", allScore[2]

        print prediction

        # for rr in prediction:
        #     if rr == 0 :
        #         context = zmq.Context()

        #         #  Socket to talk to server
        #         print("Connecting to ryu-restapi.py…")
        #         socket = context.socket(zmq.REQ)
        #         socket.connect("tcp://localhost:5555")

        #         # Sending message to Server
        #         print("Sending request …" )
        #         socket.send(b"Dattaa")

        #         #  Get the reply.
        #         message = socket.recv()
        #         print("Received reply : [ %s ]" %  message)

        print "\nDone"

    print "\n"
    pass

l = task.LoopingCall(doWork)
l.start(timeout) # call every sixty seconds

reactor.run()