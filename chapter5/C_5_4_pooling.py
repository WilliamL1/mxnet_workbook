# 5.4 池化层
## 池化层(pooling) 的提出，是为了缓解卷积层对位置的过度敏感性。

from mxnet import nd
from mxnet.gluon import nn


# 5.4.1
def pool2d(X, pool_size, mode='max'):
    h, w = pool_size
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = (X[i:i + h, j:j + w]).max()
            elif mode == 'avg':
                Y[i, j] = (X[i:i + h, j:j + w]).mean()

    return Y


def test_541():
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), mode='avg'))


def test_542():
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    X = nd.arange(16).reshape((1,1,4,4))
    print(pool2d(X))

    pool2d = nn.MaxPool2D((2,3), padding=(1,2), strides=(2,3))
    X = nd.arange(16).reshape((1,1,4,4))
    print(pool2d(X))

if __name__ == '__main__':
    # test_541()
    test_542()
