import collections
import pickle as pickle
import xml.etree.ElementTree as ET
from glob import glob
from ucca import convert, layer0

UNK = 'UNK'

class Node:
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.fprop = False
        
    def set_children_binarized(self, children):
        if len(children) == 0: # No children: leaf node
            self.isLeaf = True
        elif len(children) == 1: # One child: cut off self
            child = children[0]
            self.label = child.label
            self.word = child.word
            self.left = child.left
            self.right = child.right
            self.isLeaf = child.isLeaf
        elif len(children) == 2: # Two children: left and right
            self.left, self.right = children
            for child in children:
                child.parent = self
        else: # More than two: binarize using auxiliary node(s)
            self.left = children[0]
            self.left.parent = self
            aux = Node(self.label + '_' + children[1].label)
            self.right = aux
            self.right.parent = self
            aux.set_children_binarized(children[1:])

class Tree:

    def __init__(self,f):
        print("Reading %s..."%f)
        passage = convert.from_standard(ET.parse(f).getroot())
        self.root = Node('ROOT')
        children = [self.build(x) for l in passage.layers
                    for x in l.all if not x.incoming]
        self.root.set_children_binarized(children)

    def build(self, ucca_node):
        """
        Convert a UCCA node to a tree node along with its children
        """
        node = Node(ucca_node.tag, ucca_node.text
                    if ucca_node.layer.ID == layer0.LAYER_ID else None)
        children = [self.build(x) for x in ucca_node.children]
        node.set_children_binarized(children)
        return node

        

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
    with open('wordMap.bin','rb') as fid:
        return pickle.load(fid)

def buildWordMap(trees):
    """
    Builds map of all words in training set
    to integer values.
    """
    print("Counting words...")
    words = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)
    
    wordMap = dict(list(zip(iter(words.keys()),list(range(len(words))))))
    wordMap[UNK] = len(words) # Add unknown as word

    f = 'wordMap.bin'
    with open(f,'wb') as fid:
        pickle.dump(wordMap,fid)
    print("Wrote '%s'"%f)
        
def loadTrees(dataSet='train'):
    """
    Loads trees. Maps leaf node words to word ids.
    """
    with open('trees/%s.bin'%dataSet,'rb') as fid:
        trees = pickle.load(fid)
        
    wordMap = loadWordMap()
    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees

def buildTrees():
    """
    Loads passages and convert to trees.
    """
    trees = {}
    for dataSet in 'train', 'dev', 'test':
        passages = glob('passages/%s/*.xml' % dataSet)
        print("Reading passages in %s..."%dataSet)
        trees[dataSet] = [Tree(f) for f in passages]
        
        f = 'trees/%s.bin'%dataSet
        with open(f,'wb') as fid:
            pickle.dump(trees[dataSet],fid)
        print("Wrote '%s'"%f)
            
    buildWordMap(trees['train'])
    return trees
      
if __name__=='__main__':
    buildTrees()



