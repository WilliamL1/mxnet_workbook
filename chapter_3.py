# utf-8
# 3.1.2

import mxnet as mx
from mxnet import nd
from time import sleep

def data_faker(display = False):
    num_input = 2
    num_feature = 1000
    w = [2.5, -3.5]
    b = 0.5

    input_features = nd.random.normal(scale=1, shape=[num_feature, num_input])
    labels = w[0] * input_features[:,0] + w[1] * input_features[:,1] + b

    labels += nd.random.normal(scale=0.001, shape=labels.shape)

    if display:
        from matplotlib import pyplot as plt
        def use_svg_display():
            from IPython import display
            display.set_matplotlib_formats('svg')

        def set_figsize(figsize=(2.5, -3.5)):
            use_svg_display()
            plt.rcParams['figure.figsize'] = figsize

        set_figsize()
        # plt.scatter(input_features[:,1].asnumpy(), labels.asnumpy(),1);

    print(input_features[0])
    print(labels[0])
    return input_features, labels

def data_iter(batchsize, feature, labels, shuffle=True):
    num_feature = len(feature)
    # 随机 shuffle 的技巧是，打乱 indices ，而非 feature 本身
    indices = list(range(num_feature))
    if shuffle:
        import random
        # random 方法用的是 python random package
        random.shuffle(indices)

    for i in range(0, num_feature, batchsize):
        j = nd.array(indices[i: min(i + batchsize, num_feature)])
        # take 函数根据索引返回对应元素
        yield feature.take(j), labels.take(j)

def data_iter_test(batchsize, feature, labels, shuffle=True):
    for x, y in data_iter(batchsize, feature, labels, shuffle=True):
        print(x, y)



if __name__ == '__main__':
    # data_faker
    features, labels = data_faker()

    # data_iter
    batch_size = 25
    data_iter_test(batch_size, features, labels)

    # print(labels)