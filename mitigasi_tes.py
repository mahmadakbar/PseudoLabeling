# -*- coding: utf-8 -*-

#import zmq
#import os

# connect and open to ryuData App
# os.system("xterm -e \"python2 ryu-restapi.py\" &")
# os.system("xterm -e \"python2 cpusingle.py\" &")
# time.sleep(5)

# print "Done"
#print "Please Wait ....."

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

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import LabelEncoder
from subprocess import Popen, PIPE, STDOUT

#import matplotlib.pyplot as plt
#import seaborn as sns
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
import string

import requests
import json

#def getType(whoIP):
#    """
#    get type for calculate precision, recall, f1score
#    """
#    arrayType = []
#    for j in whoIP:
#        if j == "192.168.4.20" or j == "192.168.4.35" or j == "192.168.4.40":
#            arrayType.append('NORMAL')
#        else:
#            arrayType.append('DDOS')
#    return arrayType

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
                     "--fields=protocol,source_ip,destination_ip,honeypot",
                     "-q", """{"timestamp":{$gt:new Date(%s),$lt:new Date(%s)},$or:[{"protocol":"ICMP"},{"protocol":"TCP"},{"protocol":"UDP"}]}""" % (time2,time1),
                     "--out",
                     "output/GNB/6040/%s-data.csv" % time1])
    
    dataset = pd.read_csv('traintest/6040/dataSample/train.csv') # read & prepare data
    getDataset = pd.read_csv('output/GNB/6040/%s-data.csv' % time1)
    joinData = dataset.drop(dataset.columns[0], axis=1) # drop type
    columnz = ['serangan']
    if getDataset.shape[0]==0 :
        print "\n------------ We dont have data to be process ------------\n"
        subprocess.call("rm -r /home/lila/Documents/maynewfixlast/output/GNB/6040/%s-data.csv" % time1, shell=True)
        # subprocess.call(["rm", "output/encode/%s_encode.csv" % time1])
    else:
        print "\n----------------- data ready to process -----------------"


        Xa = joinData.drop(columns=['label'])
        Ya = joinData['label'].values

        labelencoder = LabelEncoder() # prepare for labelEncoder
        Xb = Xa.apply(labelencoder.fit_transform) # apply label encoder on "Xa"
        Xtest = getDataset.apply(labelencoder.fit_transform)
        Yb = labelencoder.fit_transform(Ya) # apply label encoder on "Ya"

        sc_X = StandardScaler() # prepare for StandardScaler
        Xtrain = sc_X.fit_transform(Xb) # apply label encoder on "X"
        Xtestaaa = sc_X.fit_transform(Xtest)

        # start execute data with ML algoritm >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        start = time.time() #timestart

        gnb = GaussianNB()
        gnb.fit(Xtrain, Yb)

        y_pred = gnb.predict(Xtestaaa)
        stop = time.time()
        timeF = stop - start
        print ("---Detecting time %s seconds ---" % timeF)
        dropPred = pd.DataFrame(y_pred, columns=columnz)
        dropPred.to_csv('output/GNB/6040/%s-hasil.csv' % time1, index=False)
        #np.savetxt("output/GNB/8020/%s-hasil.csv" % time1, y_pred, delimiter=',')

        j = 0
        k = 0
        dataMit = pd.read_csv('output/GNB/6040/%s-hasil.csv' % time1) # read & prepare data # drop type
        for i in dataMit.index:
            #print(dataset['serangan'][i])
            if dataMit['serangan'][i] == 0:
                j += 1
            else:
                k += 1
            pass

        if j > k:
            print "\n------------ DDoS Detected ------------\n"
            start2 = time.time()
            mapping = {}
            a = requests.get('http://192.168.4.9:8080/stats/switches')
            # print(a.json())
            switches = a.json()

            #get port description
            for i in switches:
                # print(i)
                command = 'http://192.168.4.9:8080/stats/portdesc/' + str(i)
                r = requests.get(command)
                temp = r.json()[str(i)]
                ports = []
                for b in temp:
                    if b['port_no'] != 'LOCAL':
                        ports.append(b['port_no'])
                        # print("DPID:"+str(i)+";Port:"+str(b['port_no']))
                mapping[i] = ports

            print(mapping)

            #mitigating - Flow Rule
            for keys, values in mapping.items():
                for a in values:
                    send = requests.post('http://192.168.4.9:8080/stats/flowentry/add', json={\
                    "dpid": keys,\
                    "cookie": 0,\
                    "table_id": 0,\
                    "idle_timeout": 60,\
                    "priority": 11111,\
                    "flags": 1,\
                    "match":{"in_port": a,"eth_type": 0x0800,"ip_proto": 1},\
                    "actions":[]\
                    })
                    print(send.status_code)

        else:
            print "\n------------ No DDoS Detected ------------\n"
        
        stop2 = time.time()
        timeF2 = stop2 - start2
        print ("--- Detecting & Mitigating time %s seconds ---" % timeF2)

        print "\nDone"

    print "\n"
    pass

l = task.LoopingCall(doWork)
l.start(timeout) # call every sixty seconds

reactor.run()
