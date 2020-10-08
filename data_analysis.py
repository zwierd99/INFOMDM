import numpy as np
import tree
import math
import sklearn
import pickle
import os.path
from mlxtend.evaluate import mcnemar_table, mcnemar

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
        self.training_x = predictor_variables
        #print(data[1:,3])
        y_to_binary = (data[1:, 3].astype(int) >= 1).astype(int)
        self.training_y = y_to_binary


        data2 = np.genfromtxt('data/eclipse-metrics-packages-3.0.csv', delimiter=';', skip_header=False, dtype=None, encoding=None)
        prerelease_bugs2 = data2[1:, 2]  # skip first element as it's the column title.
        table1_metriks2 = data2[1:, 4:44]
        prerelease_bugs2 = np.reshape(prerelease_bugs2, (-1, 1))
        predictor_variables2 = np.append(prerelease_bugs2, table1_metriks2, axis=1).astype(np.float)
        # print(predictor_variables)
        self.testing_x = predictor_variables2
        # print(data[1:,3])
        y_to_binary2 = (data2[1:, 3].astype(int) >= 1).astype(int)
        self.testing_y = y_to_binary2


class Tree:
    def __init__(self, x, y, nmin, minleaf, nfeat):
        self.tree = None
        # self.tree.name = "Tree"
        self.bag = None
        # self.bag.name = "Bag"
        self.forest = None
        # self.forest.name = "Forest"
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
                                       100)  # Needed for analysis 3
    def Analyze(self, data):
        """
        Analyzes the created trees.
        :return: Precision, accuracy and recall, and a confusion matrix for each tree.
        """
        #prediction = tree.tree_pred(data.testing_x, self.tree)
        predicted_y_tree = tree.tree_pred(data.testing_x, self.tree)
        predicted_y_bag = tree.tree_pred_b(data.testing_x, self.bag)
        predicted_y_forest = tree.tree_pred_b(data.testing_x, self.forest)

        self.mc_mota(predicted_y_tree, predicted_y_bag, data)
        self.mc_mota(predicted_y_tree, predicted_y_forest, data)
        self.mc_mota(predicted_y_bag, predicted_y_forest, data)


        # tree.print_tree(self.tree)
        open("results.txt", "w")
        for prediction in [predicted_y_tree, predicted_y_bag, predicted_y_forest]:
            print(sklearn.metrics.precision_score(data.testing_y, prediction))
            print(sklearn.metrics.recall_score(data.testing_y, prediction))
            print(sklearn.metrics.accuracy_score(data.testing_y, prediction))
            print(sklearn.metrics.confusion_matrix(data.testing_y, prediction))
            with open("results.txt", "a+") as f:
                #f.write("Showing results for " + prediction. + "\n")
                f.write("Precision: " + str(round(sklearn.metrics.precision_score(data.testing_y, prediction), 3)) + "\n")
                f.write("Recall: " + str(round(sklearn.metrics.recall_score(data.testing_y, prediction), 3)) + "\n")
                f.write("Accuracy: " + str(round(sklearn.metrics.accuracy_score(data.testing_y, prediction), 3)) + "\n")
                f.write("Confusion Matrix \n")
                f.write(np.array2string(sklearn.metrics.confusion_matrix(data.testing_y, prediction)) + "\n")
                f.write("------------------------------------------------------------\n\n")

    def mc_mota(self, y1, y2, data):
        y_target = np.array(data.testing_y)
        tb = mcnemar_table(y_target=y_target,
                           y_model1=np.array(y1),
                           y_model2=np.array(y2))

        print(tb)
        chi2, p = mcnemar(ary=tb)

        print('chi-squared:', chi2)
        print('p-value:', p)
        print()

def main():
    data = Extract()
    if not os.path.exists('trees.pkl'):
        trees = Tree(data.training_x, data.training_y, 15, 5, 41)
        f = open('trees.pkl', 'wb')
        pickle.dump(trees, f)
        f.close()
    else:
        f = open('trees.pkl', 'rb')
        trees = pickle.load(f)
        f.close()

    #print(tree.tree_pred(data.training_x, trees.tree))
    trees.Analyze(data)
    # tree.print_tree(trees.tree)


main()
