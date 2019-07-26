# 网络的初始化函数调用之后，网络数据填充之前，并不会立即对网络进行初始化，
# 原因是，输入数据的维度并不知道，无法初始化第一层权重的网络。
# 在第一次前向之后，才会进行权值初始化。

# 初始化只有在第一次前向计算时被调用，之后我们再运行前向计算net(X)时，这不会重新初始化，因此
# 不会再次产生 MyInit 实例的输出。

# 系统将真正的参数初始化延后到获得足够信息时财智星的行为叫作：延后初始化（deferred initialization）

from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(
    nn.Dense(8, activation='relu'),
    nn.Dense(10)
)

# 如果增加了 in_units=20， 就显式指定了输入节点的个数，系统可以直接推断各层网络的结构，进而
# 进而直接对权重进行了初始化。


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print(name, data.shape)


net.initialize(MyInit())

x = nd.random.uniform(shape=(2, 20))
net(x)


# x2 = nd.random.uniform(shape=(2, 30))
# net(x2)

# net.initialize(MyInit())
