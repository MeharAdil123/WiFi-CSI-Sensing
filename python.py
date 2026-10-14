import numpy as np

x= np.load("Data/UT_HAR/data/X_train.csv", allow_pickle=True)
print(x.shape)
print(x.dtype)
print(x[0])