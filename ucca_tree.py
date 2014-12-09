import collections
import pickle
import xml.etree.ElementTree as ET
import gzip
import numpy as np
from glob import glob
from io import TextIOWrapper
import sys
from ucca import convert, layer0

UNK = 'UNK'


class Node:
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.fprop = False

    def set_children_binarized(self, children):
        if len(children) == 0:  # No children: leaf node
            self.is_leaf = True
        elif len(children) == 1:  # One child: cut off self
            child = children[0]
            self.label = self.label  # + '_' + child.label
            self.word = child.word
            self.left = child.left
            self.right = child.right
            self.is_leaf = child.is_leaf
        elif len(children) == 2:  # Two children: left and right
            self.left, self.right = children
            for child in children:
                child.parent = self
        else:  # More than two: binarize using auxiliary node(s)
            self.left = children[0]
            self.left.parent = self
            aux = Node(children[1].label)  # self.label + '_' +
            self.right = aux
            self.right.parent = self
            aux.set_children_binarized(children[1:])

    def __str__(self):
        return self.word or self.label

    def subtree_str(self):
        if self.is_leaf:
            return str(self)
        else:
            return "(%s %s %s)" % (self,
                                   self.left.subtree_str(),
                                   self.right.subtree_str())

    def left_traverse(self, node_fn=None, args=None,
                      args_root=None, args_leaf=None, is_root=False):
        """
        Recursive function traverses tree
        from left to right.
        Calls node_fn at each node
        """
        if args_root is None:
            args_root = args
        if args_leaf is None:
            args_leaf = args
        node_fn(self, args_root if is_root else args_leaf if self.is_leaf else args)
        if self.left is not None:
            self.left.left_traverse(node_fn, args, args_root, args_leaf)
        if self.right is not None:
            self.right.left_traverse(node_fn, args, args_root, args_leaf)


class Tree:
    def __init__(self, f):
        if isinstance(f, Node):
            self.root = f
        else:
            print("Reading '%s'..." % f)
            passage = convert.from_standard(ET.parse(f).getroot())
            self.root = Node('ROOT')
            children = [self.build(x) for l in passage.layers
                        for x in l.all if not x.incoming]
            self.root.set_children_binarized(children)

    def build(self, ucca_node):
        """
        Convert a UCCA node to a tree node along with its children
        """
        label = get_label(ucca_node)
        if ucca_node.layer.ID == layer0.LAYER_ID:
            node = Node(label, ucca_node.text)
        else:
            node = Node(label)
        children = [self.build(x) for x in ucca_node.children]
        node.set_children_binarized(children)
        return node

    def __str__(self):
        return self.root.subtree_str()

    def left_traverse(self, node_fn=None, args=None, args_root=None, args_leaf=None):
        self.root.left_traverse(node_fn, args, args_root, args_leaf, is_root=True)


def get_label(ucca_node):
    return ucca_node.incoming[0].tag if ucca_node.incoming else 'SCENE'


def count_words(node, words):
    if node.is_leaf:
        words[node.word] += 1


def count_labels(node, labels):
    labels[node.label] += 1


def map_words(node, word_map):
    if node.is_leaf:
        node.word = word_map.get(node.word) or word_map.get(UNK)


def map_labels(node, label_map):
    node.label = label_map[node.label]


def load_word_map():
    with open('word_map.bin', 'rb') as fid:
        return pickle.load(fid)


def load_label_map():
    with open('label_map.bin', 'rb') as fid:
        return pickle.load(fid)


def build_word_map(trees, extra_words=None):
    """
    Builds map of all words in training set
    to integer values.
    If a word vector file is given, map these too
    """
    print("Counting words...")
    words = collections.defaultdict(int)
    for tree in trees:
        tree.left_traverse(node_fn=count_words, args=words)

    if extra_words is not None:
        for word in extra_words:
            words[word] += 1

    word_map = dict(list(zip(iter(words.keys()), list(range(len(words))))))
    word_map[UNK] = len(words)  # Add unknown as word

    f = 'word_map.bin'
    with open(f, 'wb') as fid:
        pickle.dump(word_map, fid)
    print("Wrote '%s'" % f)


def build_label_map(trees):
    print("Counting labels...")
    labels = collections.defaultdict(int)
    for tree in trees:
        tree.left_traverse(node_fn=count_labels, args=labels)

    labels_map = dict(list(zip(iter(labels.keys()), list(range(len(labels))))))

    f = 'label_map.bin'
    with open(f, 'wb') as fid:
        pickle.dump(labels_map, fid)
    print("Wrote '%s'" % f)


def load_word_vectors(wvec_dim, wvec_file, word_map):
    num_words = len(word_map)
    L = 0.01 * np.random.randn(wvec_dim, num_words)
    with TextIOWrapper(gzip.open(wvec_file)) as f:
        for line in f:
            fields = line.split()
            word = fields[0]
            vec = fields[1:]
            if len(vec) != wvec_dim:
                raise Exception("word vectors in %s must match wvec_dim=%d" % (wvec_file, wvec_dim))
            index = word_map.get(word, word_map[UNK])
            L[:, index] = vec
    return L


def load_trees(data_set='train'):
    """
    Loads trees. Maps leaf node words to word ids and all labels to label ids.
    """
    with open('trees/%s.bin' % data_set, 'rb') as fid:
        trees = pickle.load(fid)

    for d, fn in zip([load_word_map(), load_label_map()], [map_words, map_labels]):
        for tree in trees:
            tree.left_traverse(node_fn=fn, args=d)
    return trees


def unmap_trees(trees):
    """
    Maps leaf node words ids back to words and label ids to labels.
    """
    for d, fn in zip([load_word_map(), load_label_map()], [map_words, map_labels]):
        inverted = invert_map(d)
        for tree in trees:
            tree.left_traverse(node_fn=fn, args=inverted)
    return trees


def print_trees(f, trees, desc):
    unmap_trees(trees)
    with open(f, 'w', encoding='utf-8') as fid:
        fid.write('\n'.join([str(tree) for tree in trees]))
    print("%s trees printed to %s" % (desc, f))


def invert_map(d):
    return {v: k for k, v in d.items()}


def build_trees(wvec_file=None):
    """
    Loads passages and convert to trees.
    """
    trees = {}
    for data_set in 'train', 'dev', 'test':
        passages = glob('passages/%s/*.xml' % data_set)
        print("Reading passages in '%s'..." % data_set)
        trees[data_set] = [Tree(f) for f in passages]

        f = 'trees/%s.bin' % data_set
        with open(f, 'wb') as fid:
            pickle.dump(trees[data_set], fid)
        print("Wrote '%s'" % f)

    all_trees = [tree for t in trees.values() for tree in t]

    if wvec_file is not None:
        print("Loading words from '%s'..." % wvec_file)
        with TextIOWrapper(gzip.open(wvec_file)) as f:
            extra_words = [line.split()[0] for line in f]
    else:
        extra_words = None

    build_word_map(all_trees, extra_words)
    build_label_map(trees['train'])
    return trees


if __name__ == '__main__':
    if len(sys.argv) > 1:
        build_trees(sys.argv[1])
    else:
        build_trees()