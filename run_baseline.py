import optparse

from baseline import Baseline
from ucca_tree import *


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test", action="store_true", dest="test", default=False)
    parser.add_option("--out_file", dest="out_file", type="string",
                      default="models/baseline.bin")
    parser.add_option("--in_file", dest="in_file", type="string",
                      default="models/baseline.bin")
    parser.add_option("--data", dest="data", type="string", default="train")

    (opts, args) = parser.parse_args(args)

    # Testing
    if opts.test:
        test(opts.in_file, opts.data)
        return

    print("Loading data...")
    # load training data
    trees = load_trees()

    baseline = Baseline()
    baseline.train(trees)

    with open(opts.out_file, 'wb') as fid:
        pickle.dump(opts, fid)
        baseline.to_file(fid)


def test(baseline_file, data_set):
    assert baseline_file is not None, "Must give model to test"
    trees = load_trees(data_set)
    assert trees, "No data found"
    with open(baseline_file, 'rb') as fid:
        opts = pickle.load(fid)
        baseline = Baseline()
        baseline.from_file(fid)
    print("Testing...")
    correct, total, pred = baseline.predict(trees)
    print("Correct %d/%d, Acc %f" % (correct, total, correct / float(total)))

    print_trees('results/gold.txt', trees, 'Labeled')
    print_trees('results/pred_baseline.txt', pred, 'Baseline predicted')


if __name__ == '__main__':
    run()


