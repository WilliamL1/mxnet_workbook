"""
page: 175 ~ 182, 6 pages
time: 1 hour from 8:16am to 10:00am
"""


from d2lzh import d2l
from mxnet import nd
from mxnet.gluon import rnn

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

ctx = d2l.try_gpu()


def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                nd.zeros(num_hiddens, ctx=ctx))

    # Input gate
    W_xi, W_hi, b_i = _three()
    W_xf, W_hf, b_f = _three()
    W_xo, W_ho, b_o = _three()
    W_xc, W_hc, b_c = _three()

    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)

    return [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, b_o, W_hq, b_q]


def lstm_state_init(batch_size, num_hiddens, ctx=ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),
            nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, b_o, W_hq, b_q] = get_params()

    (H, C) = state
    outputs = []
    for X in inputs:
        f = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        i = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        o = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        c_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = f * C + i * c_tilda
        H = o * C.tanh()
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)
