import chapter3.utils as d2l
from mxnet import autograd, gluon, init, nd
import pandas as pd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.shape)
print(test_data.shape)

dt = train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]]
print(dt)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:,1:]))
print(all_features)

