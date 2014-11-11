import numpy as np
from numpy.random import sample
from collections import defaultdict

from ucca_tree import *


class Baseline:
    def __init__(self):
        self.counts = defaultdict(int)

    def train(self, trees):
        """
        Simply create a histogram of labels
        """
        for tree in trees:
            tree.left_traverse(node_fn=count_labels, args=self.counts)

    def predict(self, data):
        """
        Predict each label independently according to distribution by histogram
        """
        correct = total = 0
        trees = []

        labels = sorted(self.counts)
        probs = np.array([self.counts[label] for label in labels], dtype=float)
        probs /= probs.sum()
        bins = np.add.accumulate(probs)

        for tree in data:
            corr, tot, pred = Baseline.predict_node(tree.root, labels, bins)
            correct += corr
            total += tot
            trees.append(Tree(pred))

        return correct, total, trees

    @staticmethod
    def predict_node(node, labels, bins):
        """
        Predict label for current node and continue recursively
        """
        correct = total = 0

        if node.is_leaf:
            node.fprop = True
            left = right = None
        else:
            if not node.left.fprop:
                corr, tot, left = Baseline.predict_node(node.left, labels, bins)
                correct += corr
                total += tot
            if not node.right.fprop:
                corr, tot, right = Baseline.predict_node(node.right, labels, bins)
                correct += corr
                total += tot

        node.fprop = True

        label = labels[np.digitize(sample(1), bins)]
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
        pickle.dump(self.counts, fid)

    def from_file(self, fid):
        import pickle as pickle
        self.counts = pickle.load(fid)