import dateutil.parser as dp
import dateutil.parser
import re
import ipaddress
import socket, struct

import numpy as np
import pandas as pd
import time

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)

# ------------------ Prepare to convert Data ------------------------- #

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
    t_in_seconds = parsed_t.timestamp()
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

transfoorm = ['protocol','hpfeed_id','identifier','honeypot','type']
transfoo0rm = ['protocol','hpfeed_id','identifier','honeypot']

# ------------------- Encode data Train ----------------------- #
dataTrain = pd.read_csv('dataSplit/trainAll.csv')

dataTrain['source_ip'] = dataTrain.source_ip.apply(ip2int)
dataTrain['destination_ip'] = dataTrain.destination_ip.apply(ip2int) 
dataTrain['timestamp'] = dataTrain.timestamp.apply(iso2times)

xTrain = MultiColumnLabelEncoder(columns = transfoorm).fit_transform(dataTrain)
xTrain.to_csv(r'dataProcessing/newResult/trainAll.csv')

# ------------------- Encode data Test ----------------------- #

# Encode 10.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataTest10k = pd.read_csv('dataSplit/test10000n.csv')
TestdataID10k = pd.DataFrame(dataTest10k['_id'], columns = ['_id'])

dataTest10k['source_ip'] = dataTest10k.source_ip.apply(ip2int)
dataTest10k['destination_ip'] = dataTest10k.destination_ip.apply(ip2int) 
dataTest10k['timestamp'] = dataTest10k.timestamp.apply(iso2times)
dataTest10k = dataTest10k.drop(['_id'], axis =1)

xTest10k = MultiColumnLabelEncoder(columns = transfoo0rm).fit_transform(dataTest10k)
xTest10k.to_csv(r'dataProcessing/newResult/test10000n.csv')

# Encode 20.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataTest20k = pd.read_csv('dataSplit/test20000n.csv')
TestdataID20k = pd.DataFrame(dataTest20k['_id'], columns = ['_id'])

dataTest20k['source_ip'] = dataTest20k.source_ip.apply(ip2int)
dataTest20k['destination_ip'] = dataTest20k.destination_ip.apply(ip2int) 
dataTest20k['timestamp'] = dataTest20k.timestamp.apply(iso2times)
dataTest20k = dataTest20k.drop(['_id'], axis =1)

xTest20k = MultiColumnLabelEncoder(columns = transfoo0rm).fit_transform(dataTest20k)
xTest20k.to_csv(r'dataProcessing/newResult/test20000n.csv')

# Encode 30.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataTest30k = pd.read_csv('dataSplit/test30000n.csv')
TestdataID30k = pd.DataFrame(dataTest30k['_id'], columns = ['_id'])

dataTest30k['source_ip'] = dataTest30k.source_ip.apply(ip2int)
dataTest30k['destination_ip'] = dataTest30k.destination_ip.apply(ip2int) 
dataTest30k['timestamp'] = dataTest30k.timestamp.apply(iso2times)
dataTest30k = dataTest30k.drop(['_id'], axis =1)

xTest30k = MultiColumnLabelEncoder(columns = transfoo0rm).fit_transform(dataTest30k)
xTest30k.to_csv(r'dataProcessing/newResult/test30000n.csv')

# Encode 40.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataTest40k = pd.read_csv('dataSplit/test40000n.csv')
TestdataID40k = pd.DataFrame(dataTest40k['_id'], columns = ['_id'])

dataTest40k['source_ip'] = dataTest40k.source_ip.apply(ip2int)
dataTest40k['destination_ip'] = dataTest40k.destination_ip.apply(ip2int) 
dataTest40k['timestamp'] = dataTest40k.timestamp.apply(iso2times)
dataTest40k = dataTest40k.drop(['_id'], axis =1)

xTest40k = MultiColumnLabelEncoder(columns = transfoo0rm).fit_transform(dataTest40k)
xTest40k.to_csv(r'dataProcessing/newResult/test40000n.csv')

# Encode 50.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataTest50k = pd.read_csv('dataSplit/test50000n.csv')
TestdataID50k = pd.DataFrame(dataTest50k['_id'], columns = ['_id'])

dataTest50k['source_ip'] = dataTest50k.source_ip.apply(ip2int)
dataTest50k['destination_ip'] = dataTest50k.destination_ip.apply(ip2int) 
dataTest50k['timestamp'] = dataTest50k.timestamp.apply(iso2times)
dataTest50k = dataTest50k.drop(['_id'], axis =1)

xTest50k = MultiColumnLabelEncoder(columns = transfoo0rm).fit_transform(dataTest50k)
xTest50k.to_csv(r'dataProcessing/newResult/test50000n.csv')

# Encode 60.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataTest60k = pd.read_csv('dataSplit/test60000n.csv')
TestdataID60k = pd.DataFrame(dataTest60k['_id'], columns = ['_id'])

dataTest60k['source_ip'] = dataTest60k.source_ip.apply(ip2int)
dataTest60k['destination_ip'] = dataTest60k.destination_ip.apply(ip2int) 
dataTest60k['timestamp'] = dataTest60k.timestamp.apply(iso2times)
dataTest60k = dataTest60k.drop(['_id'], axis =1)

xTest60k = MultiColumnLabelEncoder(columns = transfoo0rm).fit_transform(dataTest60k)
xTest60k.to_csv(r'dataProcessing/newResult/test60000n.csv')

# ----------------- Training Linier SVM ---------------------- #
columns = ['protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
Y_data = xTrain['type'].values
X_data = xTrain[list(columns)].values

# 10.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>
start10k = time.time()
X_test10k = xTest10k[list(columns)].values
  
lin_clf10k = svm.LinearSVC()
lin_clf10k.fit(X_data, Y_data) 

pseudoY_test10k = lin_clf10k.predict(xTest10k)

X10k = np.vstack((X_data, X_test10k))
Y10k = np.concatenate((Y_data, pseudoY_test10k), axis=0)

pseudo_model10k = svm.LinearSVC()
pseudo_model10k.fit(X10k, Y10k)

clf10k = AdaBoostClassifier(n_estimators=500)
scores10k = cross_val_score(clf10k, X10k, Y10k)
scores10k.mean()
clf10k.fit(X10k, Y10k)

Accuracy10k = clf10k.score(X10k, Y10k)
print ("Accuracy in the training 10.000 data: ", Accuracy10k*100, "%")

stop10k = time.time()
time10k = stop10k - start10k
print("--- %s seconds ---" % time10k)

prediction10k = clf10k.predict(X_test10k)
dfPrediction10k = pd.DataFrame(data=prediction10k,columns=['type'])
dfsubmit10k = pd.concat([TestdataID10k['_id'], dfPrediction10k['type']], axis = 1, join_axes=[TestdataID10k['_id'].index])
dfsubmit10k = dfsubmit10k.reset_index(drop=True)
TestPredict10k = dfsubmit10k.to_csv('dataProcessing/newResult/resulTest10k.csv')

# 20.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>
start20k = time.time()
X_test20k = xTest20k[list(columns)].values

lin_clf20k = svm.LinearSVC()
lin_clf20k.fit(X_data, Y_data) 

pseudoY_test20k = lin_clf20k.predict(xTest20k)

X20k = np.vstack((X_data, X_test20k))
Y20k = np.concatenate((Y_data, pseudoY_test20k), axis=0)

pseudo_model20k = svm.LinearSVC()
pseudo_model20k.fit(X20k, Y20k)

clf20k = AdaBoostClassifier(n_estimators=500)
scores20k = cross_val_score(clf20k, X20k, Y20k)
scores20k.mean()
clf20k.fit(X20k, Y20k)

Accuracy20k = clf20k.score(X20k, Y20k)
print ("Accuracy in the training 20.000 data: ", Accuracy20k*100, "%")

stop20k = time.time()
time20k = stop20k - start20k
print("--- %s seconds ---" % time20k)

prediction20k = clf20k.predict(X_test20k)
dfPrediction20k = pd.DataFrame(data=prediction20k,columns=['type'])
dfsubmit20k = pd.concat([TestdataID20k['_id'], dfPrediction20k['type']], axis = 1, join_axes=[TestdataID20k['_id'].index])
dfsubmit20k = dfsubmit20k.reset_index(drop=True)
TestPredict20k = dfsubmit20k.to_csv('dataProcessing/newResult/resulTest20k.csv')

# 30.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>
start30k = time.time()
X_test30k = xTest30k[list(columns)].values

lin_clf30k = svm.LinearSVC()
lin_clf30k.fit(X_data, Y_data) 

pseudoY_test30k = lin_clf30k.predict(xTest30k)

X30k = np.vstack((X_data, X_test30k))
Y30k = np.concatenate((Y_data, pseudoY_test30k), axis=0)

pseudo_model30k = svm.LinearSVC()
pseudo_model30k.fit(X30k, Y30k)

clf30k = AdaBoostClassifier(n_estimators=500)
scores30k = cross_val_score(clf30k, X30k, Y30k)
scores30k.mean()
clf30k.fit(X30k, Y30k)

Accuracy30k = clf30k.score(X30k, Y30k)
print ("Accuracy in the training 30.000 data: ", Accuracy30k*100, "%")

stop30k = time.time()
time30k = stop30k - start30k
print("--- %s seconds ---" % time30k)

prediction30k = clf30k.predict(X_test30k)
dfPrediction30k = pd.DataFrame(data=prediction30k,columns=['type'])
dfsubmit30k = pd.concat([TestdataID30k['_id'], dfPrediction30k['type']], axis = 1, join_axes=[TestdataID30k['_id'].index])
dfsubmit30k = dfsubmit30k.reset_index(drop=True)
TestPredict30k = dfsubmit30k.to_csv('dataProcessing/newResult/resulTest30k.csv')

# 40.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>
start40k = time.time()
X_test40k = xTest40k[list(columns)].values

lin_clf40k = svm.LinearSVC()
lin_clf40k.fit(X_data, Y_data) 

pseudoY_test40k = lin_clf40k.predict(xTest40k)

X40k = np.vstack((X_data, X_test40k))
Y40k = np.concatenate((Y_data, pseudoY_test40k), axis=0)

pseudo_model40k = svm.LinearSVC()
pseudo_model40k.fit(X40k, Y40k)

clf40k = AdaBoostClassifier(n_estimators=500)
scores40k = cross_val_score(clf40k, X40k, Y40k)
scores40k.mean()
clf40k.fit(X40k, Y40k)

Accuracy40k = clf40k.score(X40k, Y40k)
print ("Accuracy in the training 40.000 data: ", Accuracy40k*100, "%")

stop40k = time.time()
time40k = stop40k - start40k
print("--- %s seconds ---" % time40k)

prediction40k = clf40k.predict(X_test10k)
dfPrediction40k = pd.DataFrame(data=prediction40k,columns=['type'])
dfsubmit40k = pd.concat([TestdataID40k['_id'], dfPrediction40k['type']], axis = 1, join_axes=[TestdataID40k['_id'].index])
dfsubmit40k = dfsubmit40k.reset_index(drop=True)
TestPredict40k = dfsubmit40k.to_csv('dataProcessing/newResult/resulTest40k.csv')

# 50.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>
start50k = time.time()
X_test50k = xTest50k[list(columns)].values

lin_clf50k = svm.LinearSVC()
lin_clf50k.fit(X_data, Y_data) 

pseudoY_test50k = lin_clf50k.predict(xTest50k)

X50k = np.vstack((X_data, X_test50k))
Y50k = np.concatenate((Y_data, pseudoY_test50k), axis=0)

pseudo_model50k = svm.LinearSVC()
pseudo_model50k.fit(X50k, Y50k)

clf50k = AdaBoostClassifier(n_estimators=500)
scores50k = cross_val_score(clf50k, X50k, Y50k)
scores50k.mean()
clf50k.fit(X50k, Y50k)

Accuracy50k = clf50k.score(X50k, Y50k)
print ("Accuracy in the training 50.000 data: ", Accuracy50k*100, "%")

stop50k = time.time()
time50k = stop50k - start50k
print("--- %s seconds ---" % time50k)

prediction50k = clf50k.predict(X_test50k)
dfPrediction50k = pd.DataFrame(data=prediction50k,columns=['type'])
dfsubmit50k = pd.concat([TestdataID50k['_id'], dfPrediction50k['type']], axis = 1, join_axes=[TestdataID50k['_id'].index])
dfsubmit50k = dfsubmit50k.reset_index(drop=True)
TestPredict50k = dfsubmit50k.to_csv('dataProcessing/newResult/resulTest50k.csv')

# 60.000 data test >>>>>>>>>>>>>>>>>>>>>>>>>>>
start60k = time.time()
X_test60k = xTest60k[list(columns)].values

lin_clf60k = svm.LinearSVC()
lin_clf60k.fit(X_data, Y_data) 

pseudoY_test60k = lin_clf60k.predict(xTest60k)

X60k = np.vstack((X_data, X_test60k))
Y60k = np.concatenate((Y_data, pseudoY_test60k), axis=0)

pseudo_model60k = svm.LinearSVC()
pseudo_model60k.fit(X60k, Y60k)

clf60k = AdaBoostClassifier(n_estimators=500)
scores60k = cross_val_score(clf60k, X60k, Y60k)
scores60k.mean()
clf60k.fit(X60k, Y60k)

Accuracy60k = clf60k.score(X60k, Y60k)
print ("Accuracy in the training 60.000 data: ", Accuracy60k*100, "%")

stop60k = time.time()
time60k = stop60k - start60k
print("--- %s seconds ---" % time60k)

prediction60k = clf10k.predict(X_test60k)
dfPrediction60k = pd.DataFrame(data=prediction60k,columns=['type'])
dfsubmit60k = pd.concat([TestdataID60k['_id'], dfPrediction60k['type']], axis = 1, join_axes=[TestdataID60k['_id'].index])
dfsubmit60k = dfsubmit60k.reset_index(drop=True)
TestPredict60k = dfsubmit60k.to_csv('dataProcessing/newResult/resulTest60k.csv')

dataPlot = {"amountData":[10000,20000,30000,40000,50000,60000], 
            "timeData":[time10k, time20k, time30k, time40k, time50k, time60k], 
            "accuracyData":[Accuracy10k, Accuracy20k, Accuracy30k, Accuracy40k, Accuracy50k, Accuracy60k] }
dfdataPlot = pd.DataFrame(dataPlot) 
plotTime = sns.catplot(x="amountData", y="timeData", data=dfdataPlot, height=5, kind="bar", palette="muted")
plotAccuracy = sns.catplot(x="amountData", y="accuracyData", data=dfdataPlot, height=5, kind="bar", palette="muted")
plt.show(plotTime)
plt.show(plotAccuracy)

print ("\nDone.")




