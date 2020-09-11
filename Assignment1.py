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