

import collections
UNK = 'UNK'

class Node:
    def __init__(self,phraseId,label=None,word=None):
        self.phraseId = phraseId
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

    def addNodes(self, nodes):
        leftTraverse(self.root,nodeFn=addNode, args=nodes)


def build_trees(f, test=False):
  nodesByPhrase = {}
  nodesById = {}
  trees = []
  print "Reading trees in %s.." % f
  with open(f,'r') as fid:
    fid.readline() # skip title
    for line in fid:
      fields = line.split("\t")
      try:
        phraseId = int(fields[0])
        node = Node(phraseId)
        node.sentenceId = int(fields[1])
        node.phrase = fields[2].strip()
        node.label = 2 if test else int(fields[3])
      except IndexError:
        raise Exception("Invalid format: '%s'" % line)
      node.tokens = node.phrase.split()
      nodesByPhrase[node.phrase.lower()] = node
      nodesById[phraseId] = node

      if node.phrase and (not trees or node.sentenceId != trees[-1].root.sentenceId):
        trees.append(Tree(node))

  for tree in trees:
    link_nodes(tree.root, nodesByPhrase)

  return trees, nodesById

def link_nodes(node, nodesByPhrase):
  tryAll = False
  num_tokens = len(node.tokens)
  for i in 2 * range(1, num_tokens):
    left, right = [nodesByPhrase.get(" ".join(tokens).lower())
                   for tokens in node.tokens[:i], node.tokens[i:]]
    if left and right and (tryAll or \
       node.phraseId + 1 in [child.phraseId for child in left, right]):
      node.left, node.right = [link_nodes(child, nodesByPhrase)
                               for child in left, right]
      node.left.parent = node.right.parent = node
      return node
    if i == num_tokens - 1: tryAll = True # go over again with no id constraint
  else:
    node.isLeaf = True
    node.word = node.phrase
    return node
  raise Exception("Failed linking node '%s'" % node.phrase)


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

def addNode(node, arg):
  if node.phraseId: arg[node.phraseId] = node

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
    trees = build_trees('data/train.tsv')
    trees += build_trees('data/test.tsv', True)

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
    trees, nodes = build_trees('data/%s.tsv'%dataSet, test)

    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees, nodes


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
    train, nodes = loadTrees()



