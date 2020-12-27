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
import string

import requests
import json

print "Please Wait ....."

def sourceipgen(dstIP, srcIP, dstPrt, srcPrt, macSrc, macDst, tyPee, coDee, chkSum, idNtfier):
    tip = dstIP
    sip = srcIP
    tpr = dstPrt
    spr = srcPrt
    ms = macSrc
    md = macDst
    tpe = tyPee
    cde = coDee
    csm = chkSum
    idf = idNtfier
    not_valid = [10, 127, 254, 255, 1, 2, 169, 172, 192]
    icmpExcp = [2,3,4,5,7]
    icmpExcp2 = randrange(44, 252)
    first = randrange(1, 256)
    # tuwooo = randrange(0, 255)
    tuwooo = randrange(0, 8)
    if dstIP == '':
        while first in not_valid:
            first = randrange(1, 256)
        tip = ".".join([str(first), str(randrange(1, 256)), str(randrange(1, 256)), str(randrange(1, 256))])
    if srcIP == '':
        while first in not_valid:
            first = randrange(1, 256)
        sip = ".".join([str(first), str(randrange(1, 256)), str(randrange(1, 256)), str(randrange(1, 256))])
    if dstPrt == '':
        tpr = random.randint(1, 1024)
    if srcPrt == '':
        spr = random.randint(1, 1024)
    if macSrc == '':
        mac = [0x00, 0x16, 0x3e,
               random.randint(0x00, 0x7f),
               random.randint(0x00, 0xff),
               random.randint(0x00, 0xff)]
        ms = ':'.join(map(lambda x: "%02x" % x, mac))
    if macDst == '':
        mac = [0x00, 0x16, 0x3e,
               random.randint(0x00, 0x7f),
               random.randint(0x00, 0xff),
               random.randint(0x00, 0xff)]
        md = ':'.join(map(lambda x: "%02x" % x, mac))
    if tyPee == '':
        hexDmp = "".join([chr(random.randint(0x00, 0xff)), chr(random.randint(0x00, 0xff)), chr(random.randint(0x00, 0xff))])
        hexDmp2 = "".join([chr(0x00),chr(0x00),chr(0x00),chr(0x00),chr(0x00),chr(0x10),chr(0x11),chr(0x12),chr(0x13),chr(0x14),chr(0x15),chr(0x16),chr(0x17),chr(0x18),chr(0x19),chr(0x1a),chr(0x1b),chr(0x1c),chr(0x1d),chr(0x1e),chr(0x1f)])
        hexDmp3 = "".join([chr(0x20),chr(0x21),chr(0x22),chr(0x23),chr(0x24),chr(0x25),chr(0x26),chr(0x27),chr(0x28),chr(0x29),chr(0x2a),chr(0x2b),chr(0x2c),chr(0x2d),chr(0x2e),chr(0x2f),chr(0x30),chr(0x31),chr(0x32),chr(0x33),chr(0x34),chr(0x35),chr(0x36),chr(0x37),])
        tpe = "".join([hexDmp, hexDmp2, hexDmp3])
    if coDee == '':
        cde = random.randint(1, 40000)
    if chkSum == '':
        csm = random.randint(1, 40000)
    if idNtfier == '':
        idf = random.randint(10000, 50000)
    return (tip, sip, tpr, spr, ms, md, tpe, cde, csm, idf)

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

def getRyu(whoRyu):
    #get DPID from switches
    mapping = {}
    a = requests.get('http://192.168.3.10:8080/stats/switches')
    # print(a.json())
    switches = a.json()

    #get port description
    for i in switches:
        # print(i)
        command = 'http://192.168.3.10:8080/stats/portdesc/' + str(i)
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
            send = requests.post('http://192.168.3.10:8080/stats/flowentry/add', json={\
            "dpid": keys,\
            "cookie": 0,\
            "table_id": 0,\
            "hard_timeout": 60,\
            "priority": 11111,\
            "flags": 1,\
            "match":{"in_port": a,"eth_type": 0x0800,"ip_proto": 1},\
            "actions":[]\
            })
            print(send.status_code)

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

        clf = AdaBoostClassifier(n_estimators=1)
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

        dstIP = ''
        srcIP = ''
        count = ''
        dstPrt = ''
        srcPrt = ''
        ptCl = ''
        macSrc = ''
        macDst = ''
        leng = ''
        tyPee = ''
        coDee = ''
        chkSum = ''
        idNtfier = ''

        ipsrc = dataset2[['source_ip', 'destination_ip']]
        pcapConverter = ipsrc.to_numpy()

        pktdump = PcapWriter("output/pcap/%s.pcap" % time1, append=True, sync=True)
        for w in pcapConverter:
            x = sourceipgen(dstIP, srcIP, dstPrt, srcPrt, macSrc, macDst, tyPee, coDee, chkSum, idNtfier)
            packets = Ether(src=x[4], dst="f0:76:1c:6e:35:94", type=0x800)/IP(id=int(x[9]) ,src=w[0], dst=w[1], flags=0x2, ttl=64, proto=1)/ICMP(type=8, code=0, id=int(x[7]), seq=int(x[8]))
            pktdump.write(packets)

        # #getpycap = "output/pcap/%s.pcap" % time1
        # #getRyu(getpycap)

        print "\nDone"

    print "\n"
    pass

l = task.LoopingCall(doWork)
l.start(timeout) # call every sixty seconds

reactor.run()