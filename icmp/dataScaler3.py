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

labelencoder = LabelEncoder()

dataset = pd.read_csv('data/new/newDatasetv3.csv') # read & prepare data
columnz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot']
columnzz = ['_id','protocol','hpfeed_id','timestamp','source_ip','destination_ip','identifier','honeypot','type']

# ------------------- Prepare and Encode Data ----------------------- #
Xa = dataset.drop(columns=['type']) # drop type
Ya = dataset['type'].values # get type values

labelencoder = LabelEncoder() # prepare for labelEncoder
Xb = Xa.apply(labelencoder.fit_transform) # apply label encoder on "Xa"
Yb = labelencoder.fit_transform(Ya) # apply label encoder on "Ya"

sc_X = StandardScaler() # prepare for StandardScaler
X = sc_X.fit_transform(Xb) # apply StandadScaler on "Xb"

# 90:10 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
train90d, test10d, Y_data90d, getType10d = train_test_split(X, Yb, test_size = 0.1)
# dropTrain = pd.DataFrame(train90d, columns=columnz) # call 'X test' array and make them to dataframe
# dropTrain.to_csv(r'dataSample/scaler/train_scaler.csv') # save to csv

# dropTest = pd.DataFrame(test10d, columns=columnz) # call 'X test' array and make them to dataframe
# dropTest.to_csv(r'dataSample/scaler/test_scaler.csv') # save to csv

# Prepare for execution
X_data90d = train90d # x train
X_test10d = test10d # x test

# Start processing data "n_estimator=10" ....................................
start10e10 = time.time() #timestart

lin_clf10e10 = svm.LinearSVC()
lin_clf10e10.fit(X_data90d, Y_data90d) 

pseudoY_test10e10 = lin_clf10e10.predict(test10d)

X10e10 = np.vstack((X_data90d, X_test10d))
Y10e10 = np.concatenate((Y_data90d, pseudoY_test10e10), axis=0)

pseudo_model10e10 = svm.LinearSVC()
pseudo_model10e10.fit(X10e10, Y10e10)

clf10e10 = AdaBoostClassifier(n_estimators=10)
scores10e10 = cross_val_score(clf10e10, X10e10, Y10e10)
scores10e10.mean()
clf10e10.fit(X10e10, Y10e10)

AccuracY10e10 = clf10e10.score(X10e10, Y10e10)
print ("Accuracy in the training 90:10 data(n_estimator=10): ", AccuracY10e10*100, "%")

stop10e10 = time.time()
time10e10 = stop10e10 - start10e10
print("--- %s seconds ---" % time10e10)

prediction10e10 = clf10e10.predict(X_test10d)
allScore10e10 = precision_recall_fscore_support(getType10d, prediction10e10, average='micro')
print("Precision    : ", allScore10e10[0])
print("Recall       : ", allScore10e10[1])
print("f1 Socre     : ", allScore10e10[2])

# dfprediction10e10 = pd.DataFrame(data=prediction10e10,columns=['type'])
# dfsubmit10e10 = pd.concat([test10d['_id'], dfprediction10e10['type']], axis = 1, join_axes=[test10d['_id'].index])
# dfsubmit10e10 = dfsubmit10e10.reset_index(drop=True)
# TestPredict10e10 = dfsubmit10e10.to_csv('dataSample/result/result90d10_e10.csv')

# Start processing data "n_estimator=50" ....................................
start10e50 = time.time() #timestart

lin_clf10e50 = svm.LinearSVC()
lin_clf10e50.fit(X_data90d, Y_data90d) 

pseudoY_test10e50 = lin_clf10e50.predict(test10d)

X10e50 = np.vstack((X_data90d, X_test10d))
Y10e50 = np.concatenate((Y_data90d, pseudoY_test10e50), axis=0)

pseudo_model10e50 = svm.LinearSVC()
pseudo_model10e50.fit(X10e50, Y10e50)

clf10e50 = AdaBoostClassifier(n_estimators=50)
scores10e50 = cross_val_score(clf10e50, X10e50, Y10e50)
scores10e50.mean()
clf10e50.fit(X10e50, Y10e50)

AccuracY10e50 = clf10e50.score(X10e50, Y10e50)
print ("Accuracy in the training 90:10 data(n_estimator=50): ", AccuracY10e50*100, "%")

stop10e50 = time.time()
time10e50 = stop10e50 - start10e50
print("--- %s seconds ---" % time10e50)

prediction10e50 = clf10e50.predict(X_test10d)
allScore10e50 = precision_recall_fscore_support(getType10d, prediction10e50, average='micro')
print("Precision    : ", allScore10e50[0])
print("Recall       : ", allScore10e50[1])
print("f1 Socre     : ", allScore10e50[2])

# dfprediction10e50 = pd.DataFrame(data=prediction10e50,columns=['type'])
# dfsubmit10e50 = pd.concat([test10d['_id'], dfprediction10e50['type']], axis = 1, join_axes=[test10d['_id'].index])
# dfsubmit10e50 = dfsubmit10e50.reset_index(drop=True)
# TestPredict10e50 = dfsubmit10e50.to_csv('dataSample/result/result90d10_e50.csv')

# Start processing data "n_estimator=100" ....................................
start10e100 = time.time() #timestart

lin_clf10e100 = svm.LinearSVC()
lin_clf10e100.fit(X_data90d, Y_data90d) 

pseudoY_test10e100 = lin_clf10e100.predict(test10d)

X10e100 = np.vstack((X_data90d, X_test10d))
Y10e100 = np.concatenate((Y_data90d, pseudoY_test10e100), axis=0)

pseudo_model10e100 = svm.LinearSVC()
pseudo_model10e100.fit(X10e100, Y10e100)

clf10e100 = AdaBoostClassifier(n_estimators=100)
scores10e100 = cross_val_score(clf10e100, X10e100, Y10e100)
scores10e100.mean()
clf10e100.fit(X10e100, Y10e100)

AccuracY10e100 = clf10e100.score(X10e100, Y10e100)
print ("Accuracy in the training 90:10 data(n_estimator=100): ", AccuracY10e100*100, "%")

stop10e100 = time.time()
time10e100 = stop10e100 - start10e100
print("--- %s seconds ---" % time10e100)

prediction10e100 = clf10e100.predict(X_test10d)
allScore10e100 = precision_recall_fscore_support(getType10d, prediction10e100, average='micro')
print("Precision    : ", allScore10e100[0])
print("Recall       : ", allScore10e100[1])
print("f1 Socre     : ", allScore10e100[2])

# dfprediction10e100 = pd.DataFrame(data=prediction10e100,columns=['type'])
# dfsubmit10e100 = pd.concat([test10d['_id'], dfprediction10e100['type']], axis = 1, join_axes=[test10d['_id'].index])
# dfsubmit10e100 = dfsubmit10e100.reset_index(drop=True)
# TestPredict10e100 = dfsubmit10e100.to_csv('dataSample/result/result90d10_e100.csv')

print("\n")
# 80:20 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
train80d, test20d, Y_data80d, getType20d = train_test_split(X, Yb, test_size = 0.2)

# Prepare for execution
X_data80d = train80d
X_test20d = test20d

# Start processing data "n_estimator=10" ....................................
start20e10 = time.time() #timestart

lin_clf20e10 = svm.LinearSVC()
lin_clf20e10.fit(X_data80d, Y_data80d) 

pseudoY_test20e10 = lin_clf20e10.predict(test20d)

X20e10 = np.vstack((X_data80d, X_test20d))
Y20e10 = np.concatenate((Y_data80d, pseudoY_test20e10), axis=0)

pseudo_model20e10 = svm.LinearSVC()
pseudo_model20e10.fit(X20e10, Y20e10)

clf20e10 = AdaBoostClassifier(n_estimators=10)
scores20e10 = cross_val_score(clf20e10, X20e10, Y20e10)
scores20e10.mean()
clf20e10.fit(X20e10, Y20e10)

AccuracY20e10 = clf20e10.score(X20e10, Y20e10)
print ("Accuracy in the training 80:20 data(n_estimator=10): ", AccuracY20e10*100, "%")

stop20e10 = time.time()
time20e10 = stop20e10 - start20e10
print("--- %s seconds ---" % time20e10)

prediction20e10 = clf20e10.predict(X_test20d)
allScore20e10 = precision_recall_fscore_support(getType20d, prediction20e10, average='micro')
print("Precision    : ", allScore20e10[0])
print("Recall       : ", allScore20e10[1])
print("f1 Socre     : ", allScore20e10[2])

# dfprediction20e10 = pd.DataFrame(data=prediction20e10,columns=['type'])
# dfsubmit20e10 = pd.concat([test20d['_id'], dfprediction20e10['type']], axis = 1, join_axes=[test20d['_id'].index])
# dfsubmit20e10 = dfsubmit20e10.reset_index(drop=True)
# TestPredict20e10 = dfsubmit20e10.to_csv('dataSample/result/result80d20_e10.csv')

# Start processing data "n_estimator=50" ....................................
start20e50 = time.time() #timestart

lin_clf20e50 = svm.LinearSVC()
lin_clf20e50.fit(X_data80d, Y_data80d) 

pseudoY_test20e50 = lin_clf20e50.predict(test20d)

X20e50 = np.vstack((X_data80d, X_test20d))
Y20e50 = np.concatenate((Y_data80d, pseudoY_test20e50), axis=0)

pseudo_model20e50 = svm.LinearSVC()
pseudo_model20e50.fit(X20e50, Y20e50)

clf20e50 = AdaBoostClassifier(n_estimators=50)
scores20e50 = cross_val_score(clf20e50, X20e50, Y20e50)
scores20e50.mean()
clf20e50.fit(X20e50, Y20e50)

AccuracY20e50 = clf20e50.score(X20e50, Y20e50)
print ("Accuracy in the training 80:20 data(n_estimator=50): ", AccuracY20e50*100, "%")

stop20e50 = time.time()
time20e50 = stop20e50 - start20e50
print("--- %s seconds ---" % time20e50)

prediction20e50 = clf20e50.predict(X_test20d)
allScore20e50 = precision_recall_fscore_support(getType20d, prediction20e50, average='micro')
print("Precision    : ", allScore20e50[0])
print("Recall       : ", allScore20e50[1])
print("f1 Socre     : ", allScore20e50[2])

# dfprediction20e50 = pd.DataFrame(data=prediction20e50,columns=['type'])
# dfsubmit20e50 = pd.concat([test20d['_id'], dfprediction20e50['type']], axis = 1, join_axes=[test20d['_id'].index])
# dfsubmit20e50 = dfsubmit20e50.reset_index(drop=True)
# TestPredict20e50 = dfsubmit20e50.to_csv('dataSample/result/result80d20_e50.csv')

# Start processing data "n_estimator=100" ....................................
start20e100 = time.time() #timestart

lin_clf20e100 = svm.LinearSVC()
lin_clf20e100.fit(X_data80d, Y_data80d) 

pseudoY_test20e100 = lin_clf20e100.predict(test20d)

X20e100 = np.vstack((X_data80d, X_test20d))
Y20e100 = np.concatenate((Y_data80d, pseudoY_test20e100), axis=0)

pseudo_model20e100 = svm.LinearSVC()
pseudo_model20e100.fit(X20e100, Y20e100)

clf20e100 = AdaBoostClassifier(n_estimators=100)
scores20e100 = cross_val_score(clf20e100, X20e100, Y20e100)
scores20e100.mean()
clf20e100.fit(X20e100, Y20e100)

AccuracY20e100 = clf20e100.score(X20e100, Y20e100)
print ("Accuracy in the training 80:20 data(n_estimator=100): ", AccuracY20e100*100, "%")

stop20e100 = time.time()
time20e100 = stop20e100 - start20e100
print("--- %s seconds ---" % time20e100)

prediction20e100 = clf20e100.predict(X_test20d)
allScore20e100 = precision_recall_fscore_support(getType20d, prediction20e100, average='micro')
print("Precision    : ", allScore20e100[0])
print("Recall       : ", allScore20e100[1])
print("f1 Socre     : ", allScore20e100[2])

# dfprediction20e100 = pd.DataFrame(data=prediction20e100,columns=['type'])
# dfsubmit20e100 = pd.concat([test20d['_id'], dfprediction20e100['type']], axis = 1, join_axes=[test20d['_id'].index])
# dfsubmit20e100 = dfsubmit20e100.reset_index(drop=True)
# TestPredict20e100 = dfsubmit20e100.to_csv('dataSample/result/result80d20_e100.csv')

print("\n")
# 70:30 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
train70d, test30d, Y_data70d, getType30d = train_test_split(X, Yb, test_size = 0.3)


# Prepare for execution
X_data70d = train70d
X_test30d = test30d

# Start processing data "n_estimator=10" ....................................
start30e10 = time.time() #timestart

lin_clf30e10 = svm.LinearSVC()
lin_clf30e10.fit(X_data70d, Y_data70d) 

pseudoY_test30e10 = lin_clf30e10.predict(test30d)

X30e10 = np.vstack((X_data70d, X_test30d))
Y30e10 = np.concatenate((Y_data70d, pseudoY_test30e10), axis=0)

pseudo_model30e10 = svm.LinearSVC()
pseudo_model30e10.fit(X30e10, Y30e10)

clf30e10 = AdaBoostClassifier(n_estimators=10)
scores30e10 = cross_val_score(clf30e10, X30e10, Y30e10)
scores30e10.mean()
clf30e10.fit(X30e10, Y30e10)

AccuracY30e10 = clf30e10.score(X30e10, Y30e10)
print ("Accuracy in the training 70:30 data(n_estimator=10): ", AccuracY30e10*100, "%")

stop30e10 = time.time()
time30e10 = stop30e10 - start30e10
print("--- %s seconds ---" % time30e10)

prediction30e10 = clf30e10.predict(X_test30d)
allScore30e10 = precision_recall_fscore_support(getType30d, prediction30e10, average='micro')
print("Precision    : ", allScore30e10[0])
print("Recall       : ", allScore30e10[1])
print("f1 Socre     : ", allScore30e10[2])

# dfprediction30e10 = pd.DataFrame(data=prediction30e10,columns=['type'])
# dfsubmit30e10 = pd.concat([test30d['_id'], dfprediction30e10['type']], axis = 1, join_axes=[test30d['_id'].index])
# dfsubmit30e10 = dfsubmit30e10.reset_index(drop=True)
# TestPredict30e10 = dfsubmit30e10.to_csv('dataSample/result/result70d30_e10.csv')

# Start processing data "n_estimator=50" ....................................
start30e50 = time.time() #timestart

lin_clf30e50 = svm.LinearSVC()
lin_clf30e50.fit(X_data70d, Y_data70d) 

pseudoY_test30e50 = lin_clf30e50.predict(test30d)

X30e50 = np.vstack((X_data70d, X_test30d))
Y30e50 = np.concatenate((Y_data70d, pseudoY_test30e50), axis=0)

pseudo_model30e50 = svm.LinearSVC()
pseudo_model30e50.fit(X30e50, Y30e50)

clf30e50 = AdaBoostClassifier(n_estimators=50)
scores30e50 = cross_val_score(clf30e50, X30e50, Y30e50)
scores30e50.mean()
clf30e50.fit(X30e50, Y30e50)

AccuracY30e50 = clf30e50.score(X30e50, Y30e50)
print ("Accuracy in the training 70:30 data(n_estimator=50): ", AccuracY30e50*100, "%")

stop30e50 = time.time()
time30e50 = stop30e50 - start30e50
print("--- %s seconds ---" % time30e50)

prediction30e50 = clf30e50.predict(X_test30d)
allScore30e50 = precision_recall_fscore_support(getType30d, prediction30e50, average='micro')
print("Precision    : ", allScore30e50[0])
print("Recall       : ", allScore30e50[1])
print("f1 Socre     : ", allScore30e50[2])

# dfprediction30e50 = pd.DataFrame(data=prediction30e50,columns=['type'])
# dfsubmit30e50 = pd.concat([test30d['_id'], dfprediction30e50['type']], axis = 1, join_axes=[test30d['_id'].index])
# dfsubmit30e50 = dfsubmit30e50.reset_index(drop=True)
# TestPredict30e50 = dfsubmit30e50.to_csv('dataSample/result/result70d30_e50.csv')

# Start processing data "n_estimator=100" ....................................
start30e100 = time.time() #timestart

lin_clf30e100 = svm.LinearSVC()
lin_clf30e100.fit(X_data70d, Y_data70d) 

pseudoY_test30e100 = lin_clf30e100.predict(test30d)

X30e100 = np.vstack((X_data70d, X_test30d))
Y30e100 = np.concatenate((Y_data70d, pseudoY_test30e100), axis=0)

pseudo_model30e100 = svm.LinearSVC()
pseudo_model30e100.fit(X30e100, Y30e100)

clf30e100 = AdaBoostClassifier(n_estimators=100)
scores30e100 = cross_val_score(clf30e100, X30e100, Y30e100)
scores30e100.mean()
clf30e100.fit(X30e100, Y30e100)

AccuracY30e100 = clf30e100.score(X30e100, Y30e100)
print ("Accuracy in the training 70:30 data(n_estimator=100): ", AccuracY30e100*100, "%")

stop30e100 = time.time()
time30e100 = stop30e100 - start30e100
print("--- %s seconds ---" % time30e100)

prediction30e100 = clf30e100.predict(X_test30d)
allScore30e100 = precision_recall_fscore_support(getType30d, prediction30e100, average='micro')
print("Precision    : ", allScore30e100[0])
print("Recall       : ", allScore30e100[1])
print("f1 Socre     : ", allScore30e100[2])

# dfprediction30e100 = pd.DataFrame(data=prediction30e100,columns=['type'])
# dfsubmit30e100 = pd.concat([test30d['_id'], dfprediction30e100['type']], axis = 1, join_axes=[test30d['_id'].index])
# dfsubmit30e100 = dfsubmit30e100.reset_index(drop=True)
# TestPredict30e100 = dfsubmit30e100.to_csv('dataSample/result/result70d30_e100.csv')

print("\n")
# 60:40 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
train60d, test40d, Y_data60d, getType40d = train_test_split(X, Yb, test_size = 0.4)

# Prepare for execution
X_data60d = train60d
X_test40d = test40d

# # Start processing data "n_estimator=10" ....................................
start40e10 = time.time() #timestart

lin_clf40e10 = svm.LinearSVC()
lin_clf40e10.fit(X_data60d, Y_data60d) 

pseudoY_test40e10 = lin_clf40e10.predict(test40d)

X40e10 = np.vstack((X_data60d, X_test40d))
Y40e10 = np.concatenate((Y_data60d, pseudoY_test40e10), axis=0)

pseudo_model40e10 = svm.LinearSVC()
pseudo_model40e10.fit(X40e10, Y40e10)

clf40e10 = AdaBoostClassifier(n_estimators=10)
scores40e10 = cross_val_score(clf40e10, X40e10, Y40e10)
scores40e10.mean()
clf40e10.fit(X40e10, Y40e10)

AccuracY40e10 = clf40e10.score(X40e10, Y40e10)
print ("Accuracy in the training 60:40 data(n_estimator=10): ", AccuracY40e10*100, "%")

stop40e10 = time.time()
time40e10 = stop40e10 - start40e10
print("--- %s seconds ---" % time40e10)

prediction40e10 = clf40e10.predict(X_test40d)
allScore40e10 = precision_recall_fscore_support(getType40d, prediction40e10, average='micro')
print("Precision    : ", allScore40e10[0])
print("Recall       : ", allScore40e10[1])
print("f1 Socre     : ", allScore40e10[2])

# dfprediction40e10 = pd.DataFrame(data=prediction40e10,columns=['type'])
# dfsubmit40e10 = pd.concat([test40d['_id'], dfprediction40e10['type']], axis = 1, join_axes=[test40d['_id'].index])
# dfsubmit40e10 = dfsubmit40e10.reset_index(drop=True)
# TestPredict40e10 = dfsubmit40e10.to_csv('dataSample/result/result60d40_e10.csv')

# Start processing data "n_estimator=50" ....................................
start40e50 = time.time() #timestart

lin_clf40e50 = svm.LinearSVC()
lin_clf40e50.fit(X_data60d, Y_data60d) 

pseudoY_test40e50 = lin_clf40e50.predict(test40d)

X40e50 = np.vstack((X_data60d, X_test40d))
Y40e50 = np.concatenate((Y_data60d, pseudoY_test40e50), axis=0)

pseudo_model40e50 = svm.LinearSVC()
pseudo_model40e50.fit(X40e50, Y40e50)

clf40e50 = AdaBoostClassifier(n_estimators=50)
scores40e50 = cross_val_score(clf40e50, X40e50, Y40e50)
scores40e50.mean()
clf40e50.fit(X40e50, Y40e50)

AccuracY40e50 = clf40e50.score(X40e50, Y40e50)
print ("Accuracy in the training 60:40 data(n_estimator=50): ", AccuracY40e50*100, "%")

stop40e50 = time.time()
time40e50 = stop40e50 - start40e50
print("--- %s seconds ---" % time40e50)

prediction40e50 = clf40e50.predict(X_test40d)
allScore40e50 = precision_recall_fscore_support(getType40d, prediction40e50, average='micro')
print("Precision    : ", allScore40e50[0])
print("Recall       : ", allScore40e50[1])
print("f1 Socre     : ", allScore40e50[2])

# dfprediction40e50 = pd.DataFrame(data=prediction40e50,columns=['type'])
# dfsubmit40e50 = pd.concat([test40d['_id'], dfprediction40e50['type']], axis = 1, join_axes=[test40d['_id'].index])
# dfsubmit40e50 = dfsubmit40e50.reset_index(drop=True)
# TestPredict40e50 = dfsubmit40e50.to_csv('dataSample/result/result60d40_e50.csv')

# Start processing data "n_estimator=100" ....................................
start40e100 = time.time() #timestart

lin_clf40e100 = svm.LinearSVC()
lin_clf40e100.fit(X_data60d, Y_data60d) 

pseudoY_test40e100 = lin_clf40e100.predict(test40d)

X40e100 = np.vstack((X_data60d, X_test40d))
Y40e100 = np.concatenate((Y_data60d, pseudoY_test40e100), axis=0)

pseudo_model40e100 = svm.LinearSVC()
pseudo_model40e100.fit(X40e100, Y40e100)

clf40e100 = AdaBoostClassifier(n_estimators=100)
scores40e100 = cross_val_score(clf40e100, X40e100, Y40e100)
scores40e100.mean()
clf40e100.fit(X40e100, Y40e100)

AccuracY40e100 = clf40e100.score(X40e100, Y40e100)
print ("Accuracy in the training 60:40 data(n_estimator=100): ", AccuracY40e100*100, "%")

stop40e100 = time.time()
time40e100 = stop40e100 - start40e100
print("--- %s seconds ---" % time40e100)

prediction40e100 = clf40e100.predict(X_test40d)
allScore40e100 = precision_recall_fscore_support(getType40d, prediction40e100, average='micro')
print("Precision    : ", allScore40e100[0])
print("Recall       : ", allScore40e100[1])
print("f1 Socre     : ", allScore40e100[2])

# dfprediction40e100 = pd.DataFrame(data=prediction40e100,columns=['type'])
# dfsubmit40e100 = pd.concat([test40d['_id'], dfprediction40e100['type']], axis = 1, join_axes=[test40d['_id'].index])
# dfsubmit40e100 = dfsubmit40e100.reset_index(drop=True)
# TestPredict40e100 = dfsubmit40e100.to_csv('dataSample/result/result60d40_e100.csv')

print("\n")
# 50:50 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
train50d, test50d, Y_data50d, getType50d = train_test_split(X, Yb, test_size = 0.5)

# Prepare for execution
X_data50d = train50d
X_test50d = test50d

# Start processing data "n_estimator=10" ....................................
start50e10 = time.time() #timestart

lin_clf50e10 = svm.LinearSVC()
lin_clf50e10.fit(X_data50d, Y_data50d) 

pseudoY_test50e10 = lin_clf50e10.predict(test50d)

X50e10 = np.vstack((X_data50d, X_test50d))
Y50e10 = np.concatenate((Y_data50d, pseudoY_test50e10), axis=0)

pseudo_model50e10 = svm.LinearSVC()
pseudo_model50e10.fit(X50e10, Y50e10)

clf50e10 = AdaBoostClassifier(n_estimators=10)
scores50e10 = cross_val_score(clf50e10, X50e10, Y50e10)
scores50e10.mean()
clf50e10.fit(X50e10, Y50e10)

AccuracY50e10 = clf50e10.score(X50e10, Y50e10)
print ("Accuracy in the training 50:50 data(n_estimator=10): ", AccuracY50e10*100, "%")

stop50e10 = time.time()
time50e10 = stop50e10 - start50e10
print("--- %s seconds ---" % time50e10)

prediction50e10 = clf50e10.predict(X_test50d)
allScore50e10 = precision_recall_fscore_support(getType50d, prediction50e10, average='micro')
print("Precision    : ", allScore50e10[0])
print("Recall       : ", allScore50e10[1])
print("f1 Socre     : ", allScore50e10[2])

# dfprediction50e10 = pd.DataFrame(data=prediction50e10,columns=['type'])
# dfsubmit50e10 = pd.concat([test50d['_id'], dfprediction50e10['type']], axis = 1, join_axes=[test50d['_id'].index])
# dfsubmit50e10 = dfsubmit50e10.reset_index(drop=True)
# TestPredict50e10 = dfsubmit50e10.to_csv('dataSample/result/result50d50_e10.csv')

# Start processing data "n_estimator=50" ....................................
start50e50 = time.time() #timestart

lin_clf50e50 = svm.LinearSVC()
lin_clf50e50.fit(X_data50d, Y_data50d) 

pseudoY_test50e50 = lin_clf50e50.predict(test50d)

X50e50 = np.vstack((X_data50d, X_test50d))
Y50e50 = np.concatenate((Y_data50d, pseudoY_test50e50), axis=0)

pseudo_model50e50 = svm.LinearSVC()
pseudo_model50e50.fit(X50e50, Y50e50)

clf50e50 = AdaBoostClassifier(n_estimators=50)
scores50e50 = cross_val_score(clf50e50, X50e50, Y50e50)
scores50e50.mean()
clf50e50.fit(X50e50, Y50e50)

AccuracY50e50 = clf50e50.score(X50e50, Y50e50)
print ("Accuracy in the training 50:50 data(n_estimator=50): ", AccuracY50e50*100, "%")

stop50e50 = time.time()
time50e50 = stop50e50 - start50e50
print("--- %s seconds ---" % time50e50)

prediction50e50 = clf50e10.predict(X_test50d)
allScore50e50 = precision_recall_fscore_support(getType50d, prediction50e50, average='micro')
print("Precision    : ", allScore50e50[0])
print("Recall       : ", allScore50e50[1])
print("f1 Socre     : ", allScore50e50[2])

# dfprediction50e50 = pd.DataFrame(data=prediction50e50,columns=['type'])
# dfsubmit50e50 = pd.concat([test50d['_id'], dfprediction50e50['type']], axis = 1, join_axes=[test50d['_id'].index])
# dfsubmit50e50 = dfsubmit50e50.reset_index(drop=True)
# TestPredict50e50 = dfsubmit50e50.to_csv('dataSample/result/result50d50_e50.csv')

# Start processing data "n_estimator=100" ....................................
start50e100 = time.time() #timestart

lin_clf50e100 = svm.LinearSVC()
lin_clf50e100.fit(X_data50d, Y_data50d) 

pseudoY_test50e100 = lin_clf50e100.predict(test50d)

X50e100 = np.vstack((X_data50d, X_test50d))
Y50e100 = np.concatenate((Y_data50d, pseudoY_test50e100), axis=0)

pseudo_model50e100 = svm.LinearSVC()
pseudo_model50e100.fit(X50e100, Y50e100)

clf50e100 = AdaBoostClassifier(n_estimators=100)
scores50e100 = cross_val_score(clf50e100, X50e100, Y50e100)
scores50e100.mean()
clf50e100.fit(X50e100, Y50e100)

AccuracY50e100 = clf50e100.score(X50e100, Y50e100)
print ("Accuracy in the training 50:50 data(n_estimator=100): ", AccuracY50e100*100, "%")

stop50e100 = time.time()
time50e100 = stop50e100 - start50e100
print("--- %s seconds ---" % time50e100)

prediction50e100 = clf50e100.predict(X_test50d)
allScore50e100 = precision_recall_fscore_support(getType50d, prediction50e100, average='micro')
print("Precision    : ", allScore50e100[0])
print("Recall       : ", allScore50e100[1])
print("f1 Socre     : ", allScore50e100[2])

# dfprediction50e100 = pd.DataFrame(data=prediction50e10,columns=['type'])
# dfsubmit50e100 = pd.concat([test50d['_id'], dfprediction50e10['type']], axis = 1, join_axes=[test50d['_id'].index])
# dfsubmit50e100 = dfsubmit50e100.reset_index(drop=True)
# TestPredict50e100 = dfsubmit50e100.to_csv('dataSample/result/result50d50_e100.csv')

print("\nDone")