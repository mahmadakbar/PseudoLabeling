import pandas as pd
import numpy as np


# ---------------- Split Normal --------------------- #

dfNormal = pd.read_csv('data/dataNormal.csv')
# df['split'] = np.random.randn(df.shape[0], 1)

mskNormal = np.random.rand(len(dfNormal)) <= 0.87

trainNormal = dfNormal[~mskNormal]
trainNormal.to_csv(r'dataSplit/trainNormal.csv')
# train.head(100)
testNormal = dfNormal[mskNormal]
testNormal.to_csv(r'dataSplit/testNormal.csv')
# test.head(100)

# ---------------- Split DDoS --------------------- #

dfDDoS = pd.read_csv('data/dataDDoS.csv')

mskDDoS = np.random.rand(len(dfDDoS)) <= 0.87

trainDDos = dfDDoS[~mskDDoS]
trainDDos.to_csv(r'dataSplit/trainDDos.csv')
# train.head(100)
testDDoS = dfDDoS[mskDDoS]
testDDoS.to_csv(r'dataSplit/testDDoS.csv')
# test.head(100)

# --------- Join All and Shuffle Row ------------ #

joinTrain = [trainNormal, trainDDos]
resultTrain = pd.concat(joinTrain)
shuffleTrain = resultTrain.sample(frac=1)
shuffleTrain.to_csv(r'dataSplit/trainAll.csv', index=False)

joinTest = [testNormal, testDDoS]
resultTest = pd.concat(joinTest)
shuffleTest = resultTest.sample(frac=1)
shuffleTest.to_csv(r'dataSplit/testAll.csv', index=False)

# --------- Split Data Test ------------ #

# 10.000 data test
msk10k = np.random.rand(len(shuffleTest)) <= 0.84
test10k = shuffleTest[~msk10k]
test10kk = test10k.drop(['type'], axis=1)
test10k.to_csv(r'dataSplit/test10000.csv', index=False)
test10kk.to_csv(r'dataSplit/test10000n.csv', index=False)

# 20.000 data test
msk20k = np.random.rand(len(shuffleTest)) <= 0.69
test20k = shuffleTest[~msk20k]
test20kk = test20k.drop(['type'], axis=1)
test20k.to_csv(r'dataSplit/test20000.csv', index=False)
test20kk.to_csv(r'dataSplit/test20000n.csv', index=False)

# 30.000 data test
msk30k = np.random.rand(len(shuffleTest)) <= 0.54
test30k = shuffleTest[~msk30k]
test30kk = test30k.drop(['type'], axis=1)
test30k.to_csv(r'dataSplit/test30000.csv', index=False)
test30kk.to_csv(r'dataSplit/test30000n.csv', index=False)

# 40.000 data test
msk40k = np.random.rand(len(shuffleTest)) <= 0.39
test40k = shuffleTest[~msk40k]
test40kk = test40k.drop(['type'], axis=1)
test40k.to_csv(r'dataSplit/test40000.csv', index=False)
test40kk.to_csv(r'dataSplit/test40000n.csv', index=False)

# 50.000 data test
msk50k = np.random.rand(len(shuffleTest)) <= 0.2
test50k = shuffleTest[~msk50k]
test50kk = test50k.drop(['type'], axis=1)
test50k.to_csv(r'dataSplit/test50000.csv', index=False)
test50kk.to_csv(r'dataSplit/test50000n.csv', index=False)

# 60.000 data test
test60kk = shuffleTest.drop(['type'], axis=1)
test60kk.to_csv(r'dataSplit/test60000n.csv', index=False)
