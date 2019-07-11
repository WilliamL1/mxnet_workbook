import mxnet
from mxnet import nd
from mxnet import autograd

def softmax(x):
    """
    y = x.exp() / x.exp().sum(axis=1, keepdims=1)
    :param x:
    :return:
    """
    x_log = x.exp()
    # keepdims 参数很重要，没有这个参数，就无法进行后续的广播。
    x_sum = nd.sum(x_log, axis=1, keepdims=1)
    return x_log / x_sum


def crossentropy(logic, label):
    return -nd.pick(logic, label).log()

def accuracy(y_hat, y):
    y_h = y_hat.argmax(axis=1)
    ax = (y_h == y.astype('float32')).mean().asscalar()
    return ax

def _364():
    X = nd.array([i for i in range(1, 14, 2)]).reshape(2, 3)
    x = nd.random.normal(shape=(2, 5))
    print('orgdata : {}'.format(x))
    x_prob = softmax(x)
    print('prob : {}'.format(x_prob))


def _365():
    y_hat = softmax(nd.random.normal(shape=(2, 3)))
    print(y_hat)
    y = nd.array([0, 2], dtype='int32')
    # nd.pick 将对应位置的 value 选出来
    y = crossentropy(y_hat, y)
    print(y)

def _366():
    y_hat = softmax(nd.random.normal(shape=(2, 3)))
    print(y_hat)
    y = nd.array([1, 2], dtype='int32')
    acc = accuracy(y_hat, y)
    print(acc)

"""
wait... 2019/07/11
def _367():
    num_epoch, lr = 5, 0.1
    def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
                  params=None, lr=None, trainer=None):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                with autograd.record():
                    y_hat = net(X)
                    l = loss(y_hat, y).sum()
                l.backward()
                if trainer is None:
"""


if __name__ == '__main__':
    _366()
