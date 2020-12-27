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

# print full numpy array
np.set_printoptions(threshold=np.inf)

dataset = pd.read_csv('data/new/newDatasetv3.csv') # read & prepare data
columnz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
columnzz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot','type']

Xa = dataset.drop(columns=['type']) # drop type
Ya = dataset['type'].values # get type values

labelencoder = LabelEncoder() # prepare for labelEncoder
Xb = Xa.apply(labelencoder.fit_transform) # apply label encoder on "Xa"
Yb = labelencoder.fit_transform(Ya) # apply label encoder on "Ya"

sc_X = StandardScaler() # prepare for StandardScaler
X = sc_X.fit_transform(Xb) # apply label encoder on "X"

# "x train"     "x test"   "y train"  "y test"        split all data(X) and type(Yb)      
trainScalern, testScalern, getYtrain, getYtest = train_test_split(X, Yb, test_size = 0.3)
dropTrain = pd.DataFrame(trainScalern, columns=columnz) # call 'X test' array and make them to dataframe
dropTrain.to_csv(r'dataSample/scaler/train_scaler.csv') # save to csv

dropTest = pd.DataFrame(testScalern, columns=columnz) # call 'X test' array and make them to dataframe
dropTest.to_csv(r'dataSample/scaler/test_scaler.csv') # save to csv

# insverse to labelencorder
frameFloat = sc_X.inverse_transform(trainScalern) # reTransform "X test" to labelEncoder
rmFloat = pd.DataFrame(frameFloat, columns=columnz).astype(int) # make "X test" array to dataframe
rmFloat.to_csv(r'dataSample/encode/train_encode.csv') # save to csv

frameFloat2 = sc_X.inverse_transform(testScalern) # reTransform "X test" to labelEncoder
rmFloat2 = pd.DataFrame(frameFloat2, columns=columnz).astype(int) # make "X test" array to dataframe
rmFloat2.to_csv(r'dataSample/encode/test_encode.csv') # save to csv

# inserve to normal
whoID = []
for j in rmFloat['_id']:
    cc = dataset.loc[dataset.index.values == j].values # search data from "rmFloat" by "_id" on same index of 'Xa'
    whoID.append(cc) # collect all data and save to array

arrayoftest = np.array(whoID).reshape(-1, 9) # reshape array every 9 value inside 1 array
trueData = pd.DataFrame(arrayoftest, columns=columnzz)
trueData.to_csv(r'dataSample/train.csv') # save to csv

whoID2 = []
for j2 in rmFloat2['_id']:
    cc2 = dataset.loc[dataset.index.values == j2].values # search data from "rmFloat" by "_id" on same index of 'Xa'
    whoID2.append(cc2) # collect all data and save to array

arrayoftest2 = np.array(whoID2).reshape(-1, 9) # reshape array every 9 value inside 1 array
trueData2 = pd.DataFrame(arrayoftest2, columns=columnzz)
trueData2.to_csv(r'dataSample/test.csv') # save to csv

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

dfprediction = pd.DataFrame(data=prediction,columns=['type'])
dfsubmit = pd.concat([dropTest['_id'], dfprediction['type']], axis = 1, join_axes=[dropTest['_id'].index])
dfsubmit = dfsubmit.reset_index(drop=True)
TestPredict = dfsubmit.to_csv('dataSample/result/result.csv')

# f = open('data/stdout.txt','w')
# print >>f, np.array(arrayType)
# f.close()
