from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
out = net(x)

print(
    net[0].weight.data(),
    net[0].bias.data(),
    net[0].weight.grad(),
    net[0].bias.grad(),
)

# display all parameters collected with nn.Sequential.add()
print(net.collect_params())

# display
print(net.collect_params('.*weight'))
print(net.collect_params('.*bias'))

net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data())

net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data())


class MyInit(init.Initializer):
    def _init_weight(self, name, arr):
        print('Init', name, arr.shape)
        arr[:] = nd.random.uniform(low=-10, high=10, shape=arr.shape)
        arr *= arr.abs() > 5


net.initialize(init=MyInit(), force_reinit=True)
print(net[0].weight.data())

net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data())

net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2,20))

print('net.params : ', net.collect_params())

net(x)

print(net[1].weight.data() == net[2].weight.data())
