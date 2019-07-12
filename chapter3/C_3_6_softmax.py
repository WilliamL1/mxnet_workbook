import mxnet
import sys
from mxnet import nd
from mxnet import autograd
from chapter3.C_3_3_linreg import sgd
from mxnet.gluon import data as gdata

num_inputs = 784
num_outputs = 10

# step 1 : define weights and bias
W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

# step 2 : get attach grad()
W.attach_grad()
b.attach_grad()

# step 3 : load dataset
batch_size = 256
trainsformer = gdata.vision.transforms.ToTensor()
num_workers = 4
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
train_iter = gdata.DataLoader(mnist_train.transform_first(trainsformer),
                              batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(trainsformer),
                             batch_size, shuffle=True, num_workers=num_workers)


# step 4: define activate function
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


# step 5: define loss function
def cross_entropy(logic, label):
    return -nd.pick(logic, label).log()


def accuracy(y_hat, y):
    y_h = y_hat.argmax(axis=1)
    ax = (y_h == y.astype('float32')).mean().asscalar()
    return ax


# step 6: define accuracy
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def _364():
    X = nd.array([i for i in range(1, 14, 2)]).reshape(2, 3)
    x = nd.random.normal(shape=(2, 5))
    print('orgdata : {}'.format(x))
    x_prob = softmax(x)
    print('prob : {}'.format(x_prob))


# step 7: construct net
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def _365():
    y_hat = softmax(nd.random.normal(shape=(2, 3)))
    print(y_hat)
    y = nd.array([0, 2], dtype='int32')
    # nd.pick 将对应位置的 value 选出来
    y = cross_entropy(y_hat, y)
    print(y)


def _366():
    y_hat = softmax(nd.random.normal(shape=(2, 3)))
    print(y_hat)
    y = nd.array([1, 2], dtype='int32')
    acc = accuracy(y_hat, y)
    print(acc)


def _367():
    num_epochs, lr = 5, 0.1

    # step 8: define train process
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
                    sgd(params, lr, batch_size)
                else:
                    trainer.step(batch_size)
                y = y.astype('float32')
                train_l_sum += l.asscalar()
                train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
                n += y.size
            test_acc = evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
                  (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


if __name__ == '__main__':
    _367()
