from Assignment_1 import tree_grow, tree_pred
import random as rd
import numpy as np


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    Grows a number M of trees with the parameters corresponding to tree_grow.
    :param m: number of trees to grow.
    :return trees: a list of trees.
    """
    trees = []
    for tree in range(0, m):
        xnew = []
        ynew = []
        for datapoint in x:
            ran = rd.randint(0, len(x)-1)
            xnew.append(x[ran])
            ynew.append(y[ran])
        trees.append(tree_grow(xnew, ynew, nmin, minleaf, nfeat))
    return trees

def tree_pred_b(trees, x):
    """
    Predicts the classification of x, given a number of trees.
    :param trees: list of trees
    :param x: data matrix that needs to be classified.
    :return y: predicted classification of x.
    """
    predictions = []
    for tree in trees:
        predictions.append(tree_pred(x, tree))

    y = []
    for column in range(0, len(predictions[0].training_y)):
        count = 0
        for row in range(0,len(predictions)):
            count += predictions[row][column]

        if float(count)/len(predictions[0].training_y) > 0.5:
            y.append(1)
        else:
            y.append(0)

    return y


def test():
    dataa = np.genfromtxt('pima.txt', delimiter=',', skip_header=False)
    number_of_features = (np.shape(dataa)[1] - 1)
    trees = tree_grow_b(dataa[:, :-1], dataa[:, -1], 20, 5, number_of_features, 5)
    preds = tree_pred_b(trees, dataa[:, :-1])
    input("gewonnen")

