import numpy as np
from numpy.random import sample

from ucca_tree import *


class Baseline:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.inner_counts = np.zeros(self.output_dim, dtype=float)
        self.root_counts = np.zeros(self.output_dim, dtype=float)
        self.leaf_counts = np.zeros(self.output_dim, dtype=float)

        self.stack = [self.inner_counts, self.root_counts, self.leaf_counts]

    def train(self, trees):
        """
        Simply create a histogram of labels
        """
        self.inner_counts, self.root_counts, self.leaf_counts = self.stack
        for tree in trees:
            tree.left_traverse(node_fn=count_labels, args=self.inner_counts,
                               args_root=self.root_counts, args_leaf=self.leaf_counts)

    def predict(self, data):
        """
        Predict each label independently according to distribution by histogram
        """
        correct = total = 0
        trees = []

        inner_bins, root_bins, leaf_bins = [np.add.accumulate(counts/counts.sum())
                                            for counts in self.stack]

        for tree in data:
            corr, tot, pred = Baseline.predict_node(tree.root, inner_bins, root_bins, leaf_bins, is_root=True)
            correct += corr
            total += tot
            trees.append(Tree(pred))

        return correct, total, trees

    @staticmethod
    def predict_node(node, inner_bins, root_bins, leaf_bins, is_root=False):
        """
        Predict label for current node and continue recursively
        """
        correct = total = 0

        if node.is_leaf:
            node.fprop = True
            left = right = None
        else:
            if not node.left.fprop:
                corr, tot, left = Baseline.predict_node(node.left, inner_bins, root_bins, leaf_bins)
                correct += corr
                total += tot
            if not node.right.fprop:
                corr, tot, right = Baseline.predict_node(node.right, inner_bins, root_bins, leaf_bins)
                correct += corr
                total += tot

        node.fprop = True

        bins = root_bins if is_root else leaf_bins if node.is_leaf else inner_bins
        label = np.digitize(sample(1), bins)[0]
        pred = Node(label)
        pred.word = node.word
        if node.is_leaf:
            pred.is_leaf = True
        else:
            pred.left = left
            pred.right = right
            left.parent = pred
            right.parent = pred

        return correct + (label == node.label), total + 1, pred

    def to_file(self, fid):
        import pickle as pickle
        pickle.dump(self.stack, fid)

    def from_file(self, fid):
        import pickle as pickle
        self.stack = pickle.load(fid)