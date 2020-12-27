# import numpy as np
# import pandas as pd
# import dateutil.parser as dp
# import dateutil.parser

# from sklearn.preprocessing import LabelEncoder

# def getType(tp):
#     """
#     get type for calculate precision, recall, f1score
#     """
#     k = []
#     for j in tp:
#         if j == "192.168.3.17" or j == "192.168.3.18" or j == "192.168.3.33":
#             k.append(1)
#         else:
#             k.append(0)
#     return k
    

# datasetsAll = pd.read_csv('icmp/data/datasets1000.csv')
# getIP = datasetsAll['source_ip']
# a = getType(getIP)
# g = pd.DataFrame(a)
# print g

import time

start = time.time()
stop = time.time()

waktu = stop - start

print waktu



