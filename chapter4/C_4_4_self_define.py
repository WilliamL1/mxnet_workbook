# 如何用 Gluon 自定义一个层，从而被重复利用。

from mxnet import gluon, nd
from mxnet.gluon import nn


# 4.4.1
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


def testCenteredLayer():
    layer = CenteredLayer()
    print(layer(nd.array([1, 2, 3, 4, 5])))

    net = nn.Sequential()
    net.add(nn.Dense(128),
            CenteredLayer())

    net.initialize()
    y = net(nd.random.uniform(shape=(2, 20)))
    print(y.mean().asscalar())


# 4.4.2
def ParameterDictTest():
    params = gluon.ParameterDict()
    params.get('params', shape=(2, 3))
    print(params)


class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units))

    def forward(self, x):
        return (nd.dot(x, self.weight.data()) + self.bias.data()).relu()


def testMyDense():
    mdense = MyDense(128, 20)
    print(mdense.params)

    mdense.initialize()
    # print(mdense(nd.random.uniform(shape=(2,20))))

    net = nn.Sequential()
    net.add(
        MyDense(units=128, in_units=20),
        MyDense(units=2, in_units=128))

    net.initialize()
    print(net(nd.random.uniform(shape=(2,20))))

if __name__ == '__main__':
    # testCenteredLayer()
    testMyDense()
    pass
