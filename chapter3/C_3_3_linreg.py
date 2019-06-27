import mxnet as mx
from mxnet import nd
from mxnet import autograd
from chapter3.C_3_2_data_iter import data_iter, data_faker, true_b, true_w

num_inputs = 2
num_examples = 1000
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()


def linreg(X, w, b):
    y = nd.dot(X, w) + b
    return y

def squared_loss(logit, labels):
    loss = (logit - labels.reshape(logit.shape)) ** 2 / 2
    return loss

def sgd(params, lr, batch_size):
    for param in params:
        # w = w - lr * △w / batch_size
        # 由于 每个 batch 损失的 l 的形状为 (batch_size, 1)，因此变量l 并不是一个标量，
        # 运行l.backward()将对l中的元素求和，得到新的变量，再求该变量有关模型参数的梯度。
        # param.grad 会对当前batch中的所有loss求和，再求梯度，因此需要除以 batch_size
        param[:] = param - lr * param.grad / batch_size


def main():
    features, labels = data_faker()

    batch_size = 10
    lr = 0.03
    num_epochs = 10
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batchsize=batch_size, feature=features, labels=labels):
            with autograd.record():
                logit = net(X, w, b)
                l = loss(logit, y)
            l.backward()
            sgd([w,b], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch * 1, train_l.mean().asnumpy()))

    print("\ntrue_w {}, w {}".format(true_w, w))
    print("\ntrue_b {}, b {}".format(true_b, b))
if __name__  == '__main__':
    main()
