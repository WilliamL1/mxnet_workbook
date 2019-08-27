from d2lzh import d2l
from d2lzh.d2l import to_onehot
import math
from mxnet import autograd, nd
from mxnet.gluon import trainer, loss as gloss
import time


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)


def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()

    return params


# 6.4.3
def init_rnn_state(batch_size, num_hiddens, ctx):
    """
    init_rnn_state 函数返回初始化的隐藏状态，它返回一个形状为(批量大小，隐藏单元个数)的值为0的NDArray组成的元组。
    使用元组是为了便于处理隐藏状态含有多个NDArray的情况。
    :param batch_size: 
    :param num_hiddens: 
    :param ctx: 
    :return: 
    """
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )


def rnn(inputs, state, params):
    """
    使用tanh作为激活函数，因为在输入分布较为均匀时，tanh函数值的均值为零
    :param inputs:
    :param state:
    :param params:
    :return:
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


X = nd.arange(10).reshape((2,5))
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X, vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

# 6.4.4
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))

    return ''.join([idx_to_char[i] for i in output])

y_pred = predict_rnn('双节棍', 10, rnn, params, init_rnn_state,
                     num_hiddens, vocab_size, ctx, idx_to_char,
                     char_to_idx)

print(y_pred)


# 6.4.5 梯度裁剪 (clip gradient)
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

