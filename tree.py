

import collections
UNK = 'UNK'

class Node:
    def __init__(self,id,label,word=None):
        self.id = id
        self.label = label 
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.fprop = False

class Tree:

    def __init__(self,root):
        self.root = root

    def printSentence(self):
        words = []
        leftTraverse(self.root,nodeFn=appendNodeString, args=words)
        print " ".join(words)

    def putIdsAndLabels(self, id2label):
        leftTraverse(self.root,nodeFn=putIdAndLabel, args=id2label)

class TreeBuilder:
    def __init__(self, test=False):
      self.phrase2label = {}
      self.phrase2id = {}
      self.sentences = []
      self.test = test

    def build_trees(self, f):
      last_sentence_id = None
      print "Reading trees in %s.." % f
      with open(f,'r') as fid:
        fid.readline()
        for line in fid:
          values = line.split()
          phrase_id = int(values[0])
          sentence_id = int(values[1])
          if self.test:
            phrase = values[2:]
            label = None
          else:
            phrase = values[2:-1]
            label = int(values[-1])

          flat = flatten(phrase)
          self.phrase2label[flat] = label
          self.phrase2id[flat] = phrase_id
          if sentence_id != last_sentence_id and phrase:
            last_sentence_id = sentence_id
            self.sentences.append(phrase)

      return [Tree(self.build_node(s)) for s in self.sentences]

    def build_node(self, phrase):
      flat = flatten(phrase)
      node = Node(self.phrase2id.get(flat, None),
                  self.phrase2label.get(flat, 2))
      if len(phrase) == 1:
        node.isLeaf = True
        node.word = phrase[0]
      else:
        for i in range(1, len(phrase)):
          left, right = phrase[:i], phrase[i:]
          if flatten(left) in self.phrase2label or len(right) == 1:
            node.left = self.build_node(left)
            node.right = self.build_node(right)
            node.left.parent = node.right.parent = node
            break
      assert node.isLeaf or node.left, "%s %s %s" % (phrase, left, right)
      return node

def flatten(phrase):
  return str([c for w in phrase for c in w])

        

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)

def appendNodeString(node, arg):
  if node.isLeaf: arg.append(node.word)

def putIdAndLabel(node, arg):
  if node.id: arg[node.id] = node.label

def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]

def loadWordMap():
    import cPickle as pickle
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """
    import cPickle as pickle
    trees = TreeBuilder().build_trees('data/train.tsv')
    trees += TreeBuilder(True).build_trees('data/test.tsv')

    print "Counting words.."
    words = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)
    
    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word

    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

def loadTrees(dataSet='train', test=False):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    file = 'data/%s.tsv'%dataSet
    trees = TreeBuilder(test).build_trees(file)

    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees


def unmapTrees(trees):
    """
    Maps leaf node words ids back to words and label ids to labels.
    """
    reverseWordMap = {v: k for k, v in loadWordMap().items()}
    for tree in trees:
        leftTraverse(tree.root, nodeFn=mapWords, args=reverseWordMap)
    return trees

      
if __name__=='__main__':
    buildWordMap()
    train = loadTrees()



