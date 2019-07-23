r"""

在对隐藏层使用丢弃法，假设其中 h2， h5被清零，那么输出值的计算不再依赖于 h2, h5，在反向传播时，与这两个隐藏层相关的权重的梯度均为0.
由于在训练中，每个神经元都有概率被清零，因此输出层的计算无法过度依赖于h1,...,h5中的任意一个，从而在训练模型时起到正则化的作用，并可
能用力应对过拟合。在测试模型时，我们为了得到更加确定的结果，一般不使用丢弃法。
"""

from d2lzh import d2l
from mxnet.gluon import loss as gloss, data as gdata, nn
from mxnet import autograd, nd, init, gluon


def dropout(X, drop_prob):
    r'''
    公式：
    h_i' = (ξ_i / (1 - drop_prob)) * h_i
    其中 ξ_i 是丢弃神经元的概率，设 ξ_i 为 0 和 1 的概率分别为 p 和 1-p。

    期望算法：
    E(h_i') = (E(ξ_i) / (1 - drop_prob)) * E(h_i)
            = ((1 - drop_prob) /  (1 - drop_prob)) * E(h_i)
            = E(h_i)
    期望不变

    :param X:
    :param drop_prob:
    :return:
    '''
    assert 0 < drop_prob < 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return nd.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    X_final = mask * X / keep_prob
    # print('mask : {}'.format(mask))
    return X_final


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]


def net(X):
    net = nn.Sequential()
    net = net.add(
        nn.Dense(256, activation='relu'),
        nn.Dropout(0.2),
        nn.Dense(256, activation='relu'),
        nn.Dropout(0.2),
        nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    return net


def train():
    num_epochs, lr, batch_size = 5, 0.5, 256
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
                  params, lr)


def dropout_test():
    X = nd.arange(16).reshape((2, 8))
    X_d = dropout(X, 0.2)
    print('X_d : {}'.format(X_d))
    X_d = dropout(X, 0.5)
    print('X_d : {}'.format(X_d))
    X_d = dropout(X, 0.8)
    print('X_d : {}'.format(X_d))


if __name__ == '__main__':
    train()
