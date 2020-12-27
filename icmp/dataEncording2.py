import numpy as np
import pandas as pd
import time

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

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)

labelencoder = LabelEncoder()

# Encode 50.000 data test
dataTrain50k = pd.read_csv('dataSample/train90d.csv')
xTrain50k = dataTrain50k.apply(labelencoder.fit_transform)

xTrain50k.to_csv(r'dataProcessing/train90d.csv')

# Encode 30.000 data test
dataTest30k = pd.read_csv('dataSample/test10d.csv')
xTest30k = dataTest30k.apply(labelencoder.fit_transform).drop(['type'], axis =1)

xTest30k.to_csv(r'dataProcessing/test10d.csv')

# ------------------------------------------------------------------------- #
columns = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']

Y_data50k = xTrain50k['type'].values
X_data50k = xTrain50k[list(columns)].values
X_test30k = xTest30k[list(columns)].values
# ----------------- Training Linier SVM ---------------------- #

#  30.000 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
start30k = time.time()

lin_clf30k = svm.LinearSVC()
lin_clf30k.fit(X_data50k, Y_data50k) 

pseudoY_test30k = lin_clf30k.predict(xTest30k)

X30k = np.vstack((X_data50k, X_test30k))
Y30k = np.concatenate((Y_data50k, pseudoY_test30k), axis=0)

pseudo_model30k = svm.LinearSVC()
pseudo_model30k.fit(X30k, Y30k)

clf30k = AdaBoostClassifier(n_estimators=10)
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
dfsubmit30k = pd.concat([xTest30k['_id'], dfPrediction30k['type']], axis = 1, join_axes=[xTest30k['_id'].index])
dfsubmit30k = dfsubmit30k.reset_index(drop=True)
TestPredict30k = dfsubmit30k.to_csv('dataProcessing/resulTest30k.csv')


# datasetsAll = pd.read_csv('data/datasetsallFinal.csv')
# train, test = train_test_split(datasetsAll, test_size=0.5)

# train.to_csv(r'dataSample/train90d.csv')
# test.to_csv(r'dataSample/test10d.csv')

# xTrain = train.apply(labelencoder.fit_transform)
# xTest10k = test.apply(labelencoder.fit_transform).drop(['type'], axis =1)

# xTrain.to_csv(r'dataSample/encode/train90d.csv')
# xTest10k.to_csv(r'dataSample/encode/test10d.csv')

# # --------------------------------------------------------------------- #
# columns = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
# Y_data = xTrain['type'].values
# X_data = xTrain[list(columns)].values
# X_test10k = xTest10k[list(columns)].values
# # ----------------- Training Linier SVM ---------------------- #

# #  10.000 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# start10k = time.time()
# # X_test10k = xTest10k[list(columns)].values

# lin_clf10k = svm.LinearSVC()
# lin_clf10k.fit(X_data, Y_data) 

# pseudoY_test10k = lin_clf10k.predict(xTest10k)

# X10k = np.vstack((X_data, X_test10k))
# Y10k = np.concatenate((Y_data, pseudoY_test10k), axis=0)

# pseudo_model10k = svm.LinearSVC()
# pseudo_model10k.fit(X10k, Y10k)

# clf10k = AdaBoostClassifier(n_estimators=700)
# scores10k = cross_val_score(clf10k, X10k, Y10k)
# scores10k.mean()
# clf10k.fit(X10k, Y10k)

# Accuracy10k = clf10k.score(X10k, Y10k)
# print ("Accuracy in the training 10.000 data: ", Accuracy10k*100, "%")

# stop10k = time.time()
# time10k = stop10k - start10k
# print("--- %s seconds ---" % time10k)

# prediction10k = clf10k.predict(X_test10k)
# dfPrediction10k = pd.DataFrame(data=prediction10k,columns=['type'])
# dfsubmit10k = pd.concat([xTest10k['_id'], dfPrediction10k['type']], axis = 1, join_axes=[xTest10k['_id'].index])
# dfsubmit10k = dfsubmit10k.reset_index(drop=True)
# TestPredict10k = dfsubmit10k.to_csv('dataSample/result/result90d10_e10.csv')
print("\nDone.")