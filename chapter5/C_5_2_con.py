from mxnet.gluon import nn
from mxnet import nd


# 5.2.1
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


X = nd.random.uniform(shape=(8, 8))
conv2d = nn.Conv2D(1, kernel_size=(3, 3), padding=1)
cs = comp_conv2d(conv2d, X).shape
print(cs)
