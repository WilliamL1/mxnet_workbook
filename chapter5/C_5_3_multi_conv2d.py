from mxnet import nd
import numpy as np
from d2lzh import d2l


def corr2d(X, K):
    c, h, w, = K.shape
    Y = nd.zeros((X.shape[1] - h + 1, X.shape[2] - w + 1))
    for i in range(c):
        for j in range(Y.shape[0]):
            for k in range(Y.shape[1]):
                Y[j, k] += (X[i, j:j + h, k:k + w] * K[i, :, :]).sum()
    return Y


def corr2d_multi_in(X, K):
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


def corr2d_multi_in_out(X, K):
    # k shape : c_o * c_i * k_h * k_w
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


def corr2d_multi_in_out_1x1(X, K):
    c_o = K.shape[0]
    c_i, h, w = X.shape
    X = X.reshape((c_i, -1))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == '__main__':
    x = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    k = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    # print(corr2d(x, k))
    print(corr2d_multi_in(x, k))

    K = nd.stack(k, k + 1, k + 2)
    print(K.shape)
    print(corr2d_multi_in_out(x, K))

    # test corr2d_multi_in_out_1x1
    X = nd.random.uniform(shape=(3, 3, 3))
    K = nd.random.uniform(shape=(2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)

    print((Y1 - Y2).norm().asscalar() < 1e-6)
