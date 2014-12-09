import optparse
import pickle

import sgd as optimizer
import time

from ucca_tree import *
import rnn
import rntn


models = {"rnn": rnn.RNN, "rntn": rntn.RNTN}


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test", action="store_true", dest="test", default=False)
    parser.add_option("--distance", action="store_true", dest="distance", default=False)
    parser.add_option("--metric", dest="metric", default="cosine")

    # Optimizer
    parser.add_option("--minibatch", dest="minibatch", type="int", default=30)
    parser.add_option("--optimizer", dest="optimizer", type="string",
                      default="adagrad")
    parser.add_option("--model", dest="model", type="string", default="rnn")
    parser.add_option("--epochs", dest="epochs", type="int", default=50)
    parser.add_option("--step", dest="step", type="float", default=1e-2)

    parser.add_option("--output_dim", dest="output_dim", type="int", default=0)
    parser.add_option("--wvec_dim", dest="wvec_dim", type="int", default=50)
    parser.add_option("--out_file", dest="out_file", type="string",
                      default="models/test.bin")
    parser.add_option("--in_file", dest="in_file", type="string",
                      default="models/test.bin")
    parser.add_option("--data", dest="data", type="string", default="train")
    parser.add_option("--wvec_file", dest="wvec_file", type="string", default=None)

    (opts, args) = parser.parse_args(args)

    # Testing
    if opts.test:
        test(opts.in_file, opts.data)
        return

    # Finding nearest neighbors to input words
    if opts.distance:
        distance(opts.in_file, opts.metric)
        return

    print("Loading data...")
    # load training data
    trees = load_trees()
    word_map = load_word_map()
    opts.num_words = len(word_map)
    if opts.output_dim == 0:
        opts.output_dim = len(load_label_map())

    if opts.wvec_file is None:
        wvecs = None
    else:
        print("Loading word vectors...")
        wvecs = load_word_vectors(opts.wvec_dim, opts.wvec_file, word_map)

    model = models[opts.model]
    net = model(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch, wvecs)

    sgd = optimizer.SGD(net, alpha=opts.step, minibatch=opts.minibatch,
                        optimizer=opts.optimizer)
    save(net, opts, sgd)

    for e in range(opts.epochs):
        start = time.time()
        print("Running epoch %d" % e)
        sgd.run(trees)
        end = time.time()
        print("Time per epoch : %f" % (end - start))
        save(net, opts, sgd)


def save(net, opts, sgd):
    with open(opts.out_file, 'wb') as fid:
        pickle.dump(opts, fid)
        pickle.dump(sgd.costt, fid)
        net.to_file(fid)


def test(net_file, data_set):
    trees = load_trees(data_set)
    assert trees, "No data found"
    net = load(net_file)
    print("Testing...")
    cost, correct, total, pred = net.cost_and_grad(trees, test=True)
    print("Cost %f, Correct %d/%d, Acc %f" % (cost, correct, total, correct / float(total)))

    print_trees('results/gold.txt', trees, 'Labeled')
    print_trees('results/pred.txt', pred, 'Predicted')


def distance(net_file, metric):
    net = load(net_file)
    word_map = load_word_map()
    inverted = invert_map(word_map)
    k = 10
    while True:
        try:
            word = str(input("Enter word: "))
        except EOFError: break
        index = word_map.get(word) or word_map.get(UNK)
        neighbors, distances = net.nearest(index, k, metric)
        neighbors = [inverted[index] for index in neighbors]
        print("\n".join("%-30s%.5f" % (n, d) for n, d in zip(neighbors, distances)))
    print()


def load(net_file):
    assert net_file is not None, "Must give model to test"
    with open(net_file, 'rb') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        model = models[getattr(opts, "model", net_file.split("/")[-1].partition("_")[0])]
        net = model(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch)
        net.from_file(fid)
    return net


if __name__ == '__main__':
    run()
