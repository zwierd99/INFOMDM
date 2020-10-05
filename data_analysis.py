import numpy as np
import tree
import math

class Extract:
    def __init__(self, file_path='data/eclipse-metrics-packages-2.0.csv'):
        """
        Extracts the data from the csv file into a usable array.

        :param file_path: path of the csv file, uses Version 2.0 by default.
        """
        data = np.genfromtxt(file_path, delimiter=';', skip_header=False, dtype=None, encoding=None)
        prerelease_bugs = data[1:,2] # skip first element as it's the column title.
        table1_metriks = data[1:,4:44]
        prerelease_bugs = np.reshape(prerelease_bugs, (-1, 1))
        predictor_variables = np.append(prerelease_bugs, table1_metriks, axis=1).astype(np.float)
        #print(predictor_variables)
        self.x = predictor_variables
        #print(data[1:,3])
        self.y = data[1:,3].astype(np.int)

class Tree:
    def __init__(self, x, y, nmin, minleaf, nfeat):
        self.tree = None
        self.bag = None
        self.forest = None
        self.train_trees(x, y, nmin, minleaf, nfeat)

    def train_trees(self, x, y, nmin, minleaf, nfeat):
        """
        Trains all classification trees required for part 2 of the assignment.

        :param x: data matrix containing attribute values
        :param y: vector of class labels
        :param nmin: number of minimal observations needed for splitting
        :param minleaf: minimal number of observations remaining in a node after splitting
        :param nfeat: number of features that should be considered for each split
        """
        self.tree = tree.tree_grow(x, y, nmin, minleaf, nfeat)  # Needed for analysis 1
        self.bag = tree.tree_grow_b(x, y, nmin, minleaf, nfeat, 100)  # Needed for analysis 2
        self.forest = tree.tree_grow_b(x, y, nmin, minleaf, math.floor(math.sqrt(nfeat)),
                                       100)  # Needed for analysis 3 (forest != bagging maar idk??)
    def Analyze(self):
        """
        Analyzes the created trees.
        :return: Precision, accuracy and recall, and a confusion matrix for each tree.
        """


def main():
    data = Extract()
    trees = Tree(data.x, data.y, 15, 5, 41)
    trees.Analyze()



main()
