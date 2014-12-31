import numpy as np
import collections
from scipy.spatial.distance import cdist
from ucca_tree import Node, Tree, copy_leaves, merge

np.seterr(over='raise', under='raise')


class RNN (object):
    def __init__(self, wvec_dim, output_dim, num_words, mb_size=30, wvecs=None, rho=1e-4):
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
        self.W = 0.01 * np.random.randn(self.wvec_dim, 2 * self.wvec_dim)
        self.b = np.zeros(self.wvec_dim)

        # Score weights
        self.Ws = 0.01 * np.random.randn(1, self.wvec_dim)
        self.bs = np.zeros(1)

        # Label weights for softmax
        self.Wl = 0.01 * np.random.randn(self.output_dim, self.wvec_dim)
        self.bl = np.zeros(self.output_dim)

        self.stack = [self.L, self.W, self.b, self.Ws, self.Wl, self.bl]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty(self.wvec_dim)
        self.dWs = np.empty(self.Wl.shape)
        self.dbs = np.empty(1)
        self.dWl = np.empty(self.Wl.shape)
        self.dbl = np.empty(self.output_dim)


    def cost_and_grad(self, mb_data, test=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        If test is False, returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.
        Otherwise, returns
           average cost
           # correct
           # total
           predicted trees
        If test is "labels", predict only the tree labels
        If test is "structure", predict their structure too
        """

        self.init_cost_and_grad()

        correct, cost, total, trees = self.batch_forward_prop(mb_data, test)
        if test:
            return (1. / len(mb_data)) * cost, correct, total, trees

        self.batch_back_prop(mb_data)

        # scale cost and grad by mb size
        scale = (1. / self.mb_size)
        for v in self.dL.values():
            v *= scale

        cost = self.regularize(cost)

        return scale * cost, self.grad(scale)


    def init_cost_and_grad(self):
        self.L, self.W, self.b, self.Ws, self.bs, self.Wl, self.bl = self.stack
        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dWl[:] = 0
        self.dbl[:] = 0
        self.dL = collections.defaultdict(self.default_vec)



    def batch_forward_prop(self, mb_data, test):
        # Forward prop each tree in minibatch
        cost = correct = total = 0.0
        trees = []
        for tree in mb_data:
            c, corr, tot, pred = self.forward_prop(tree, test)
            cost += c
            correct += corr
            total += tot
            if test:
                trees.append(Tree(pred))
        return correct, cost, total, trees


    def batch_back_prop(self, mb_data):
        # Back prop each tree in minibatch
        for tree in mb_data:
            self.back_prop(tree.root)


    def regularize(self, cost):
        # Add L2 Regularization
        cost += (self.rho / 2) * np.sum(self.W ** 2)
        cost += (self.rho / 2) * np.sum(self.Ws ** 2)
        cost += (self.rho / 2) * np.sum(self.Wl ** 2)
        return cost


    def grad(self, scale):
        return [
            self.dL,
            scale * (self.dW + self.rho * self.W),
            scale * self.db,
            scale * (self.dWs + self.rho * self.Ws),
            scale * self.dbs,
            scale * (self.dWl + self.rho * self.Wl),
            scale * self.dbl
        ]


    def forward_prop(self, node, test=None):
        if test == "structure":
            return self.forward_prop_structure(node.leaves())
        else:
            return self.forward_prop_labels(node.root, test)


    def forward_prop_structure(self, leaves):

        for node in leaves:
            node.h_acts = self.L[:, node.word]

        nodes = leaves[:] # copy list
        while True:
            if len(nodes) == 1:
                i = 0
            else:
                parents = [self.merge(pair) for pair in zip(nodes[:-1], nodes[1:])]
                i = np.argmax([parent.score for parent in parents])
                nodes[i] = parents[i] # replace left child with parent
                del nodes[i + 1] # remove right child
            self.softmax(nodes[i])


        cost = correct = total = 0.0

        children = []
        pred = None
        if len(leaves) == 1: # got to the root
            node = leaves[0]
            self.hidden_forward_prop(node)


        node.fprop = True

        if test:
            pred = Node(np.argmax(node.probs))
            pred.word = node.word
            if node.is_leaf:
                pred.is_leaf = True
            else:
                pred.set_children_binarized(children)

        return cost - np.log(node.probs[node.label]),\
               correct + (np.argmax(node.probs) == node.label),\
               total + 1,\
               pred


    def merge(self, nodes):
        # Merge nodes and score the potential parent
        parent = Node(None)
        parent.set_children_binarized(nodes)
        self.hidden_forward_prop(parent)
        parent.score = self.Ws.dot(parent.h_acts) + self.bs
        return parent


    def forward_prop_labels(self, node, test=None):

        cost = correct = total = 0.0

        children = []
        pred = None
        if node.is_leaf:
            node.h_acts = self.L[:, node.word]
        else: # predict labels
            for child in (node.left, node.right):
                if not child.fprop:
                    c, corr, tot, pred = self.forward_prop(child, test)
                    cost += c
                    correct += corr
                    total += tot
                    children.append(pred)
            self.hidden_forward_prop(node)

        self.softmax(node)

        node.fprop = True

        if test:
            pred = Node(np.argmax(node.probs))
            pred.word = node.word
            if node.is_leaf:
                pred.is_leaf = True
            else:
                pred.set_children_binarized(children)

        return cost - np.log(node.probs[node.label]),\
               correct + (np.argmax(node.probs) == node.label),\
               total + 1,\
               pred


    def hidden_forward_prop(self, node):
        # Affine
        lr = np.hstack([node.left.h_acts, node.right.h_acts])
        node.h_acts = np.dot(self.W, lr) + self.b
        # Relu
        # node.h_acts[node.h_acts < 0] = 0
        # Tanh
        node.h_acts = np.tanh(node.h_acts)


    def softmax(self, node):
        # Softmax for label probabilities
        node.probs = np.dot(self.Wl, node.h_acts) + self.bl
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs /= np.sum(node.probs)


    def back_prop(self, node, error=None):

        # Clear nodes
        node.fprop = False

        # Softmax grad
        deltas = node.probs
        deltas[node.label] -= 1.0
        self.dWl += np.outer(deltas, node.h_acts)
        self.dbl += deltas
        deltas = np.dot(self.Wl.T, deltas)

        if error is not None:
            deltas += error

        # deltas *= (node.h_acts != 0)
        deltas *= (1 - node.h_acts ** 2) # (tanh)' = 1-tanh^2

        # Leaf nodes update word vecs
        if node.is_leaf:
            self.dL[node.word] += deltas
            return

        # Hidden grad
        if not node.is_leaf:
            self.hidden_back_prop(deltas, node)


    def hidden_back_prop(self, deltas, node):
        self.dW += np.outer(deltas,
                            np.hstack([node.left.h_acts,
                                       node.right.h_acts]))
        self.db += deltas
        # Error signal to children
        deltas = np.dot(self.W.T, deltas)
        self.back_prop(node.left, deltas[:self.wvec_dim])
        self.back_prop(node.right, deltas[self.wvec_dim:])


    def update_params(self, scale, update, log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P, dP in zip(self.stack[1:], update[1:]):
                p_rms = np.sqrt(np.mean(P ** 2))
                dp_rms = np.sqrt(np.mean((scale * dP) ** 2))
                print("weight rms=%f -- update rms=%f" % (p_rms, dp_rms))

        self.stack[1:] = [P + scale * dP for P, dP in zip(self.stack[1:], update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.keys():
            self.L[:, j] += scale * dL[j]

    def to_file(self, fid):
        import pickle as pickle
        pickle.dump(self.stack, fid)

    def from_file(self, fid):
        import pickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self, data, epsilon=1e-6):
        cost, grad = self.cost_and_grad(data)

        for W, dW in zip(self.stack[1:], grad[1:]):
            W = W[..., None]  # add dimension since bias is flat
            dW = dW[..., None]
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i, j] += epsilon
                    cost_p, _ = self.cost_and_grad(data)
                    W[i, j] -= epsilon
                    num_grad = (cost_p - cost) / epsilon
                    err = np.abs(dW[i, j] - num_grad)
                    print("Analytic %.9f, Numerical %.9f, Relative Error %.9f" % (dW[i, j], num_grad, err))

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


    def nearest(self, word, k=10, metric='cosine'):
        self.L = self.stack[0]
        distances = cdist(self.L.T, self.L[np.newaxis, :, word], metric).reshape(-1)
        neighbors = distances.argsort()[1:k+1]
        return neighbors, distances[neighbors]


if __name__ == '__main__':
    import ucca_tree

    train = ucca_tree.load_trees()
    num_words = len(ucca_tree.load_word_map())
    output_dim = len(ucca_tree.load_label_map())
    wvec_dim = 10

    rnn = RNN(wvec_dim, output_dim, num_words, mb_size=4)

    print("Numerical gradient check...")
    rnn.check_grad(train[:4])
