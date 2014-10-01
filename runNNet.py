import optparse
import pickle

import sgd as optimizer
import rntn as nnet
import time

from uccatree import *


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test", action="store_true", dest="test", default=False)

    # Optimizer
    parser.add_option("--minibatch", dest="minibatch", type="int", default=30)
    parser.add_option("--optimizer", dest="optimizer", type="string",
                      default="adagrad")
    parser.add_option("--epochs", dest="epochs", type="int", default=50)
    parser.add_option("--step", dest="step", type="float", default=1e-2)

    parser.add_option("--outputDim", dest="outputDim", type="int", default=0)
    parser.add_option("--wvecDim", dest="wvecDim", type="int", default=30)
    parser.add_option("--outFile", dest="outFile", type="string",
                      default="models/test.bin")
    parser.add_option("--inFile", dest="inFile", type="string",
                      default="models/test.bin")
    parser.add_option("--data", dest="data", type="string", default="train")

    (opts, args) = parser.parse_args(args)

    # Testing
    if opts.test:
        test(opts.inFile, opts.data)
        return

    print("Loading data...")
    # load training data
    trees = loadTrees()
    opts.numWords = len(loadWordMap())
    if opts.outputDim == 0:
        opts.outputDim = len(loadLabelMap())

    rnn = nnet.RNN(opts.wvecDim, opts.outputDim, opts.numWords, opts.minibatch)

    sgd = optimizer.SGD(rnn, alpha=opts.step, minibatch=opts.minibatch,
                        optimizer=opts.optimizer)

    for e in range(opts.epochs):
        start = time.time()
        print("Running epoch %d" % e)
        sgd.run(trees)
        end = time.time()
        print("Time per epoch : %f" % (end - start))

        with open(opts.outFile, 'wb') as fid:
            pickle.dump(opts, fid)
            pickle.dump(sgd.costt, fid)
            rnn.toFile(fid)


def test(netFile, dataSet):
    trees = loadTrees(dataSet)
    assert netFile is not None, "Must give model to test"
    with open(netFile, 'rb') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        rnn = nnet.RNN(opts.wvecDim, opts.outputDim, opts.numWords, opts.minibatch)
        rnn.fromFile(fid)
    print("Testing...")
    cost, correct, total, pred = rnn.costAndGrad(trees, test=True, retTrees=True)
    print("Cost %f, Correct %d/%d, Acc %f" % (cost, correct, total, correct / float(total)))
    unmapTrees(trees)
    unmapTrees(pred)
    print()
    print("Correct trees:")
    for tree in trees:
        print(tree)
    print()
    print("Predicted trees:")
    for tree in pred:
        print(tree)


if __name__ == '__main__':
    run()


