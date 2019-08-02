from mxnet import nd
from mxnet.gluon import nn


# 5.1.1
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros([X.shape[0] - h + 1, X.shape[1] - w + 1])
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 5.1.2
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


# 5.1.3

def test_513():
    # 边缘检测小例
    X = nd.ones((8, 8))
    X[2:6, 2:6] = 0
    print(X)
    '''
    原始图片，任务是进行边缘检测
        [
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 0. 0. 0. 0. 1. 1.]
         [1. 1. 0. 0. 0. 0. 1. 1.]
         [1. 1. 0. 0. 0. 0. 1. 1.]
         [1. 1. 0. 0. 0. 0. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
        ]
     '''
    # 构造kernel
    K1 = nd.array([[1, -1], [1, -1]])
    Y1 = corr2d(X, K1)
    print(Y1)
    '''
    检测纵向边缘    
    [[ 0.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0. -1.  0.]
     [ 0.  2.  0.  0.  0. -2.  0.]
     [ 0.  2.  0.  0.  0. -2.  0.]
     [ 0.  2.  0.  0.  0. -2.  0.]
     [ 0.  1.  0.  0.  0. -1.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.]]
     '''
    # 构造kernel
    K2 = nd.array([[1, 1], [-1, -1]])
    Y2 = corr2d(X, K2)
    print(Y2)
    '''
    检测横向边缘    
    [[ 0.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  2.  2.  2.  1.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.]
     [ 0. -1. -2. -2. -2. -1.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.]]
    
    '''
    Y3 = nd.add(nd.abs(Y1), nd.abs(Y2))
    print(Y3)
    '''
    对两个通道做池化
    [[0. 0. 0. 0. 0. 0. 0.]
     [0. 2. 2. 2. 2. 2. 0.]
     [0. 2. 0. 0. 0. 2. 0.]
     [0. 2. 0. 0. 0. 2. 0.]
     [0. 2. 0. 0. 0. 2. 0.]
     [0. 2. 2. 2. 2. 2. 0.]
     [0. 0. 0. 0. 0. 0. 0.]]
    '''


def test_512():
    x = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    k = nd.array([[0, 1], [2, 3]])
    y = corr2d(x, k)
    print(y)

    C = Conv2D(k.shape)
    C.initialize()
    print(C(x))

    # 给weight赋值
    C2 = Conv2D(k.shape)
    C2.initialize()
    C2.weight.data()[:] = k
    print(C2(x))


def test_514():
    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)

    print(Y)

    # X = nd.random.uniform(shape=(1, 1, 6, 8))
    # Y = nd.random.uniform(shape=(1, 1, 6, 7))

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        from mxnet import autograd
        with autograd.record():
            Y_pred = conv2d(X)
            l = (Y_pred - Y) ** 2
        l.backward()
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
        if (i + 1) % 2 == 0:
            print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))
        '''
        batch 2, loss 4.949
        batch 4, loss 0.831
        batch 6, loss 0.140
        batch 8, loss 0.024
        batch 10, loss 0.004
        '''

    print(conv2d.weight.data().reshape((1,2)))
    '''
    [[ 0.9895    -0.9873705]]
    <NDArray 1x2 @cpu(0)>        
    '''

    # 几个遗留问题
    # 1. 调大 epoch 数之后，会发生过拟合，loss会变得非常大
    # 2. 如果把卷积核做成 (2,2)，会发现基本无法收敛。


if __name__ == '__main__':
    test_512()
    # test_513()
    # test_514()
