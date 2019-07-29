from mxnet import nd
from mxnet.gluon import nn


# 4.5.1
def save():
    x = nd.ones(3)
    nd.save('C_4_5_x', x)


def load():
    x = nd.load('C_4_5_x')
    print(x)


def save2():
    x = nd.ones(3)
    y = nd.zeros(4)
    dt = {
        'x': x,
        'y': y
    }
    nd.save('C_4_5_xy', [x, y])
    nd.save('C_4_5_dt', dt)

    x2, y2 = nd.load('C_4_5_xy')
    print(x2, y2)

    dt2 = nd.load('C_4_5_dt')
    print(dt2)


# 4.5.2

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    def forward(self, x):
        return self.out(self.hidden(x))


def testMLP():
    net = MLP()
    net.initialize()
    x = nd.random.uniform(shape=(2, 20))
    out = net(x)
    print(out)

    # 保存参数
    filename = 'C_4_5_mlp.params'
    net.save_parameters(filename)

    # 重新实例化，并导入参数
    net2 = MLP()
    net2.load_parameters(filename)
    out2 = net2(x)

    # 比较结果
    print(out2 == out)


if __name__ == '__main__':
    # save2()
    testMLP()
