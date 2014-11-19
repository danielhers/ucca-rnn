import optparse
import pickle
import importlib

import sgd as optimizer
import time

from ucca_tree import *


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test", action="store_true", dest="test", default=False)

    # Optimizer
    parser.add_option("--minibatch", dest="minibatch", type="int", default=30)
    parser.add_option("--optimizer", dest="optimizer", type="string",
                      default="adagrad")
    parser.add_option("--model", dest="model", type="string", default="rntn")
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

    nnet = importlib.import_module(opts.model)
    rnn = nnet.RNN(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch, wvecs)

    sgd = optimizer.SGD(rnn, alpha=opts.step, minibatch=opts.minibatch,
                        optimizer=opts.optimizer)

    for e in range(opts.epochs):
        start = time.time()
        print("Running epoch %d" % e)
        sgd.run(trees)
        end = time.time()
        print("Time per epoch : %f" % (end - start))

        with open(opts.out_file, 'wb') as fid:
            pickle.dump(opts, fid)
            pickle.dump(sgd.costt, fid)
            rnn.to_file(fid)


def test(net_file, data_set):
    assert net_file is not None, "Must give model to test"
    trees = load_trees(data_set)
    assert trees, "No data found"
    with open(net_file, 'rb') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        try:
            nnet = importlib.import_module(opts.model)
        except AttributeError:
            import rnn as nnet
        rnn = nnet.RNN(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch)
        rnn.from_file(fid)
    print("Testing...")
    cost, correct, total, pred = rnn.cost_and_grad(trees, test=True, ret_trees=True)
    print("Cost %f, Correct %d/%d, Acc %f" % (cost, correct, total, correct / float(total)))

    print_trees('results/gold.txt', trees, 'Labeled')
    print_trees('results/pred.txt', pred, 'Predicted')


if __name__ == '__main__':
    run()


