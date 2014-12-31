import numpy as np
import collections
from ucca_tree import Node, Tree
from rnn import RNN

np.seterr(over='raise', under='raise')


class RNTN (RNN):
    def __init__(self, wvec_dim, output_dim, num_words, mb_size=30, wvecs=None, rho=1e-6):
        self.wvec_dim = wvec_dim
        self.output_dim = output_dim
        self.num_words = num_words
        self.mb_size = mb_size
        self.default_vec = lambda: np.zeros((wvec_dim,))
        self.rho = rho

        # Word vectors
        if wvecs is not None:
            self.L = wvecs
        else:
            self.L = 0.01 * np.random.randn(self.wvec_dim, self.num_words)

        # Hidden activation weights
        self.V = 0.01 * np.random.randn(self.wvec_dim, 2 * self.wvec_dim, 2 * self.wvec_dim)
        self.W = 0.01 * np.random.randn(self.wvec_dim, self.wvec_dim * 2)
        self.b = np.zeros(self.wvec_dim)

        # Softmax weights
        self.Wl = 0.01 * np.random.randn(self.output_dim, self.wvec_dim)
        self.bl = np.zeros(self.output_dim)

        self.stack = [self.L, self.V, self.W, self.b, self.Wl, self.bl]

        # Gradients
        self.dV = np.empty((self.wvec_dim, 2 * self.wvec_dim, 2 * self.wvec_dim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty(self.wvec_dim)
        self.dWl = np.empty(self.Wl.shape)
        self.dbl = np.empty(self.output_dim)


    def init_cost_and_grad(self):
        self.L, self.V, self.W, self.b, self.Wl, self.bl = self.stack
        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWl[:] = 0
        self.dbl[:] = 0
        self.dL = collections.defaultdict(self.default_vec)


    def regularize(self, cost):
        # Add L2 Regularization 
        cost += (self.rho / 2) * np.sum(self.V ** 2)
        cost += (self.rho / 2) * np.sum(self.W ** 2)
        cost += (self.rho / 2) * np.sum(self.Wl ** 2)


    def grad(self, scale):
        return [
            self.dL,
            scale * (self.dV + self.rho * self.V),
            scale * (self.dW + self.rho * self.W),
            scale * self.db,
            scale * (self.dWl + self.rho * self.Wl),
            scale * self.dbl
        ]


    def hidden_forward_prop(self, node):
        # Affine
        lr = np.hstack([node.left.h_acts, node.right.h_acts])
        node.h_acts = np.dot(self.W, lr) + self.b
        node.h_acts += np.tensordot(self.V, np.outer(lr, lr), axes=([1, 2], [0, 1]))
        # Tanh
        node.h_acts = np.tanh(node.h_acts)


    def hidden_back_prop(self, deltas, node):
        lr = np.hstack([node.left.h_acts, node.right.h_acts])
        outer = np.outer(deltas, lr)
        self.dV += (np.outer(lr, lr)[..., None] * deltas).T
        self.dW += outer
        self.db += deltas
        # Error signal to children
        deltas = np.dot(self.W.T, deltas)
        deltas += np.tensordot(self.V.transpose((0, 2, 1)) + self.V,
                               outer.T, axes=([1, 0], [0, 1]))
        self.back_prop(node.left, deltas[:self.wvec_dim])
        self.back_prop(node.right, deltas[self.wvec_dim:])


    def check_grad(self, data, epsilon=1e-6):

        cost, grad = self.cost_and_grad(data)

        for W, dW in zip(self.stack[1:], grad[1:]):
            W = W[..., None, None]  # add dimension since bias is flat
            dW = dW[..., None, None]
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    for k in range(W.shape[2]):
                        W[i, j, k] += epsilon
                        cost_p, _ = self.cost_and_grad(data)
                        W[i, j, k] -= epsilon
                        num_grad = (cost_p - cost) / epsilon
                        err = np.abs(dW[i, j, k] - num_grad)
                        print("Analytic %.9f, Numerical %.9f, Relative Error %.9f" % (dW[i, j, k], num_grad, err))

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        for j in dL.keys():
            for i in range(L.shape[0]):
                L[i, j] += epsilon
                cost_p, _ = self.cost_and_grad(data)
                L[i, j] -= epsilon
                num_grad = (cost_p - cost) / epsilon
                err = np.abs(dL[j][i] - num_grad)
                print("Analytic %.9f, Numerical %.9f, Relative Error %.9f" % (dL[j][i], num_grad, err))


if __name__ == '__main__':
    import ucca_tree

    train = ucca_tree.load_trees()
    num_words = len(ucca_tree.load_word_map())
    output_dim = len(ucca_tree.load_label_map())
    wvec_dim = 10

    rntn = RNTN(wvec_dim, output_dim, num_words, mb_size=4)

    print("Numerical gradient check...")
    rntn.check_grad(train[:1])
