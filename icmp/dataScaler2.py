import numpy as np
import pandas as pd
import time

from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns

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

# print full numpy array
np.set_printoptions(threshold=np.inf)

dataset = pd.read_csv('dataSample/test.csv') # read & prepare data
dataset2 = pd.read_csv('data/tetes.csv')

columnz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
columnzz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot','type']
columnzzz = ['_id','protocol','hpfeed_id','source_ip','destination_ip','identifier','honeypot']

dropUnsuable = dataset.drop(dataset.columns[0], axis=1) # drop type

joinData = pd.concat([dropUnsuable, dataset2], ignore_index=True)
# joinData.to_csv(r'dataSample/testes.csv')

Xa = joinData.drop(columns=['type'])
Ya = joinData['type'].values

labelencoder = LabelEncoder() # prepare for labelEncoder
Xb = Xa.apply(labelencoder.fit_transform) # apply label encoder on "Xa"
Yb = labelencoder.fit_transform(Ya) # apply label encoder on "Ya"

sc_X = StandardScaler() # prepare for StandardScaler
X = sc_X.fit_transform(Xb) # apply label encoder on "X"

scalerDf = pd.DataFrame(X, columns=columnz)

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

# print arrayTrain

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

clf = AdaBoostClassifier(n_estimators=4)
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

pktdump = PcapWriter("test.pcap", append=True, sync=True)
for w in pcapConverter:
    x = sourceipgen(dstIP, srcIP, dstPrt, srcPrt, macSrc, macDst, tyPee, coDee, chkSum, idNtfier)
    packets = Ether(src=x[4], dst="f0:76:1c:6e:35:94", type=0x800)/IP(id=int(x[9]) ,src=w[0], dst=w[1], flags=0x2, ttl=64, proto=1)/ICMP(type=8, code=0, id=int(x[7]), seq=int(x[8]))
    pktdump.write(packets)

print "\nDone"

# dfprediction = pd.DataFrame(data=prediction,columns=['type'])
# dfsubmit = pd.concat([dropTest['_id'], dfprediction['type']], axis = 1, join_axes=[dropTest['_id'].index])
# dfsubmit = dfsubmit.reset_index(drop=True)
# TestPredict = dfsubmit.to_csv('dataSample/result/result.csv')

# f = open('data/stdout.txt','w')
# print >>f, getYtrain
# f.close()
