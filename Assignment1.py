import numpy as np
class Node:
    '''
    Defines the Node structure used in the functions below.
    '''
    def __init__(self, value=None, children=[], parent=None, split=None):
        self.value = value
        self.children = children
        self.parent = parent
        self.split = split

class Tree:
    '''
    Defines the Tree structure used in the functions below.
    '''
    def __init__(self, root=Node()):
        self.root = root


    def print_tree(self):
        print(self.root.value)
        childrenstr = ""
        for child in self.children:
            childrenstr += child.value + " "
        print(childrenstr)

def tree_grow(x, y, nmin, minleaf, nfeat):
    '''
    Grows a classification tree for dataset x, given the labels y.
    
    :param x: 2 dimensional data matrix.
    :param y: class label vector.
    :param nmin: number of observations that a node must contain at least, for it to be allowed to be split.
    :param minleaf: minimum number of observations required for a leafnode.
    :param nfeat: number of features thatshould be considered for each split.
    :return: a tree object that can be used for predicting new cases.
    '''
    i = 0
    trainingdata = []
    for point in x:

        trainingdata.append({"data":x[i],
                             "label":y[i]})
        i += 1
    print(trainingdata)
    nodelist = trainingdata

    sorteddata = {data: label for data, label in sorted(trainingdata[0].items(), key=lambda item: item[0])}
    print(sorteddata)
    while len(nodelist) != 0:
        #currentnode = Node(nodelist.pop())
        #print(currentnode.value[""])
        if impurity(currentnode) > 0:
            candidatesplits = y
            for i in range(len(currentnode.value[0])):
                h =1


def split(x, i):
    return (x[0:i], x[i:])

def impurity(node):
    '''
    Calculates the impurity of a node.
    '''
    p = np.sum(node.value[1])/len(node.value[1])
    return p * (1 - p)

def tree_pred(x, tr):
    '''
    Predicts the labels for a dataset x, given a tree tr.
    
    :param x: 2 dimensional data matrix, containing the attribute values of the cases for which predictions are required.
    :param tr: tree  object  created  with  the  function tree_grow.
    :return: the vector y of predicted class labels for the cases in x.
    '''

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
   '''
   Grows a number m trees.
   
   :param m: denotes the number of bootstrap samples to be drawn.
   :return: a list trList of m trees.
   '''


def tree_pred_b(x, trList):
    '''
    Predicts the labels for a dataset x, given a tree tr.

    :param x: 2 dimensional data matrix, containing the attribute values of the cases for which predictions are required.
    :param trList: list of tree  objects  created  with  the  function tree_grow.
    :return: the vector y of predicted class labels for the cases in x. The predicted class label is obtainted by
    taking the majority vote of the prediction given by every tree in the list trList.
    '''
tree_grow([[1,5,3],[4,2,6]],[0,1],0,0,0)
