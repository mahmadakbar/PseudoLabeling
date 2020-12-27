import numpy as np
import pandas as pd

import sched
import time
import subprocess
import sys
import dateutil.parser as dp
import dateutil.parser
import re
import ipaddress
import socket, struct
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
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)

def getType(whoIP):
    """
    get type for calculate precision, recall, f1score
    """
    arrayType = []
    for j in whoIP:
        if j == "192.168.3.17" or j == "192.168.3.18" or j == "192.168.3.33":
            arrayType.append(1)
        else:
            arrayType.append(0)
    return arrayType

def ip2int(ip):
    """
    Convert an IP string to int
    """
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]

def iso2times(t):
    """
    Convert an iso 8601 to unix timestamp
    """
    parsed_t = dp.parse(t)
    t_in_seconds = parsed_t.strftime("%s")
    x = re.sub('\.', '', str(t_in_seconds)) 
    return x

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X) 

timeout = 10.0 # one seconds

labelencoder = LabelEncoder()

trainDatasets = pd.read_csv('icmp/data/datasets6000n.csv')

columns = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
transfoorm = ['_id','protocol','hpfeed_id','destination_ip','identifier','honeypot','type']
transfoorm2 = ['_id','protocol','hpfeed_id','destination_ip','identifier','honeypot']

# train
trainDatasets['source_ip'] = trainDatasets.source_ip.apply(ip2int)
trainDatasets['timestamp'] = trainDatasets.timestamp.apply(iso2times)

encodeTrain = MultiColumnLabelEncoder(columns = transfoorm).fit_transform(trainDatasets)
encodeTrain.to_csv(r'output/encode/trainDatasets.csv')

def do_something():
    ts = time.time()
    g = str(ts).replace('.', '')
    subprocess.check_output(["mongoexport",
                             "--db", 
                             "mnemosyne", 
                             "--collection", 
                             "session", "--type=csv", 
                             "--fields", 
                             "_id,protocol,hpfeed_id,timestamp,source_ip,destination_ip,identifier,honeypot", 
                             "--out",
                             "output/%s.csv" % g])
    k1 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k1.communicate(input="db.session.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k2 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k2.communicate(input="db.metadata.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k3 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k3.communicate(input="db.counts.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k4 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k4.communicate(input="db.file.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k5 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k5.communicate(input="db.hpfeed.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k6 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k6.communicate(input="db.dork.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k7 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k7.communicate(input="db.url.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k8 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k8.communicate(input="db.daily_stats.remove({})")
    subprocess.Popen("exit 1", shell=True)

    testDatasets = pd.read_csv('output/%s.csv' % g)
    getIP = testDatasets['source_ip']
    aType = getType(getIP)
    trueType = pd.DataFrame(aType)

    # test
    testDatasets['source_ip'] = testDatasets.source_ip.apply(ip2int)
    testDatasets['timestamp'] = testDatasets.timestamp.apply(iso2times)
    
    encodeTest = MultiColumnLabelEncoder(columns = transfoorm2).fit_transform(testDatasets)
    encodeTest.to_csv(r'output/encode/%s_encode.csv' % g)

    # Prepare for execution
    Y_data = encodeTrain['type'].values
    X_data = encodeTrain[list(columns)].values
    X_test = encodeTest[list(columns)].values

    if not aType :
        print "\n------------ We dont have data to be process ------------\n"
        subprocess.call(["rm", "output/%s.csv" % g])
        subprocess.call(["rm", "output/encode/%s_encode.csv" % g])
    else:
        print "\n----------------- data ready to process -----------------"
        # Start processing data "n_estimator=10" ....................................
        start = time.time() #timestart

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_data, Y_data) 

        pseudoY_test = lin_clf.predict(encodeTest)

        X = np.vstack((X_data, X_test))
        Y = np.concatenate((Y_data, pseudoY_test), axis=0)

        pseudo_model = svm.LinearSVC()
        pseudo_model.fit(X, Y)

        clf = AdaBoostClassifier(n_estimators=2)
        scores = cross_val_score(clf, X, Y)
        scores.mean()
        clf.fit(X, Y)

        AccuracY = clf.score(X, Y)
        print "Accuracy :", AccuracY*100, "%"

        stop = time.time()
        timeF = stop- start
        print("--- %s seconds ---" % timeF)

        prediction = clf.predict(X_test)
        allScore = precision_recall_fscore_support(trueType, prediction, average='micro')
        print "Precision    : ", allScore[0]
        print "Recall       : ", allScore[1]
        print "f1 Socre     : ", allScore[2]

        prediction = clf.predict(X_test)
        dfPrediction = pd.DataFrame(data=prediction,columns=['type'])
        dfsubmit = pd.concat([encodeTest['_id'], dfPrediction['type']], axis = 1, join_axes=[encodeTest['_id'].index])
        dfsubmit = dfsubmit.reset_index(drop=True)
        TestPredict = dfsubmit.to_csv('output/result/%s_result.csv'% g)
    print "\n------------------------- Done -------------------------\n"
    pass

l = task.LoopingCall(do_something)
l.start(timeout) # call every one seconds

reactor.run()
