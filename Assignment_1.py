import numpy as np
import random as rd
import math
from sklearn.metrics import confusion_matrix


class tree_grow:
    def __init__(self, x, y, nmin, minleaf, nfeat):
        """
        Grow a decision tree on data x with labels y, according to parameters nmin, minleaf and nfeat
        :param x: data matrix containing attribute values
        :param y: vector of class labels
        :param nmin: number of minimal observations needed for splitting
        :param minleaf: minimal number of observations remaining in a node after splitting
        :param nfeat: number of features that should be considered for each split
        """
        # Merge the data (x) and the respective labels(y) into a numpy array and add to the nodelist
        x = np.array(x)
        y = np.expand_dims(np.transpose(np.array(y)), axis=1)

        xy_combined = np.concatenate([x, y], axis=1)

        xy_node = node(xy_combined, 0, False)
        nodelist = [xy_node]

        # Bool to check if we are in root to build up tree
        root = True

        # Initialize whole tree
        tree = 0

        self.min_leaf = minleaf

        # While there are still element left in the nodelist perform this
        while nodelist:
            # Shuffle the nodelist to pick a random current_node each time
            rd.shuffle(nodelist)
            current_node = nodelist.pop()

            # set root node to original node
            if root:
                tree = current_node
                root = False

            current_impurity = self.impurity(current_node.data)
            if current_impurity > 0 and np.shape(current_node.data)[0] >= nmin:

                # Obtain the indices of the features that are considered
                feature_set_indices = self.draw_features(current_node.data[:, :-1], nfeat)

                # Obtain values of best possible split from features selected above
                impurity, split_col, split_point = self.select_feature(current_node.data, feature_set_indices)
                impurity_reduction = current_impurity - impurity  # die ook ergens meegeven??

                # Satisfy minleaf constraint
                if split_point != -1:
                    left_node, right_node = self.select_children(current_node, split_col, split_point)

                    nodelist.append(left_node)
                    nodelist.append(right_node)

                    current_node.left_child = left_node
                    current_node.right_child = right_node

                    current_node.split_value = split_point
                    current_node.split_col = split_col
        self.tree = tree

    def give_tree(self):
        """
        Return built up tree
        :return: tree
        """
        return self.print_tree(self.tree)

    def impurity(self, current_node):
        """
        Calculate impurity of current node
        :param current_node:
        :return:
        """
        p = np.sum(current_node[:, -1]) / current_node[:, -1].shape[0]
        return p * (1 - p)

    def draw_features(self, x, nfeat):
        """
        Obtain random set of columns for possible splits
        :param x: data with multiple columns/features
        :param nfeat: amount of features that need to be selected
        :return: returns set of features available for possible splits
        """
        nr_columns = x.shape[1]
        column_indices = rd.sample(range(0, nr_columns), nfeat)
        return column_indices

    def select_feature(self, current_node, feature_set_indices):
        """
        Decide best possible split from selected columns
        :param current_node: Data and labels belonging to the current node
        :param feature_set_indices: indices of the columns which will be compared to eachother
        :return: returns values belonging to the best possible split
        """
        x = current_node[:, :-1]
        y = current_node[:, -1]

        lowest_impurity = math.inf
        best_col_index = -1
        best_split_point = -1

        # Calculate impurity minimization for each column and choose lowest impurity
        for col in feature_set_indices:
            split_point, impurity = self.best_split(x[:, col], y)
            if impurity <= lowest_impurity:
                lowest_impurity = impurity
                best_col_index = col
                best_split_point = split_point

        return lowest_impurity, best_col_index, best_split_point

    def best_split(self, data, labels):
        """
        Computes best split value on a designated feature
        :param data:    numeric values
        :param labels:  labels (0/1) of data
        :return:    returns data value of the best possible split position
        """
        # Sort both lists in the same order
        data, labels = zip(*sorted(zip(data, labels)))

        # Remove duplicate values
        sorted_data_unique = np.array(sorted(set(data)))
        labels = np.array(labels)

        data_len = len(sorted_data_unique)
        split_points = (sorted_data_unique[0:data_len - 1] + sorted_data_unique[1:data_len]) / 2
        step_size = 1 / len(data)

        lowest_impurity = 1
        best_split_point = -1
        for split in split_points:  # See classification trees - 1, slide 31 for more info
            left_list = labels[data <= split]
            ratio_left = step_size * len(left_list)
            p = self.chance_of(0, left_list)
            left_eq = ratio_left * p * (1 - p)

            right_list = labels[data > split]
            ratio_right = 1 - ratio_left
            q = self.chance_of(0, right_list)
            right_eq = ratio_right * q * (1 - q)

            current_impurity = left_eq + right_eq
            if len(left_list) >= self.min_leaf and len(
                    right_list) >= self.min_leaf and lowest_impurity >= current_impurity:
                lowest_impurity = current_impurity
                best_split_point = split

        return best_split_point, lowest_impurity

    def chance_of(self, x, litty):  # LITTY
        """
        Chance of certain element x in a list
        :param x:   wanted value
        :param litty:   list to search
        :return:    occurences of x in litty
        """
        count = 0
        for i in litty:
            if i == x:
                count += 1
        return count / len(litty)

    def select_children(self, current_node, split_col, split_point):
        """
        Select children given the split points on the selected column
        :param current_node:    parent node on which the split is performed
        :param split_col:   index of column on which split is performed
        :param split_point:     value of splitting point in the selected column
        :return: return left and right child nodes
        """
        # Determine splits according to the split point
        left_child = current_node.data[current_node.data[:, split_col] <= split_point]
        right_child = current_node.data[current_node.data[:, split_col] > split_point]

        # Transform raw data children to node children
        left_node = node(left_child, current_node.depth + 1, False)
        right_node = node(right_child, current_node.depth + 1, True)

        return left_node, right_node

    def print_tree(self, current_node):
        """
        Print tree in ordered structure, works recursively
        :param current_node: current node
        :return: prints tree
        """
        if current_node.left_child:
            print(" |\t" * current_node.depth, "|--- feature", current_node.split_col, "<=", current_node.split_value)
            self.print_tree(current_node.left_child)
            print(" |\t" * current_node.depth, "|--- feature", current_node.split_col, ">", current_node.split_value)
            self.print_tree(current_node.right_child)
        else:
            print(" |\t" * (current_node.depth), "|---", self.majority(current_node))

    def majority(self, current_node):
        """
        Calculate the majority label of the current node
        :param current_node: node of which we want the majority label
        :return: majority label (0/1)
        """
        labels = current_node.data[:, -1]
        zero_sum = (labels == 0).sum()
        one_sum = (labels == 1).sum()

        if zero_sum >= one_sum:
            return 0
        else:
            return 1


class tree_pred:
    def __init__(self, x, tr):
        """
        returns predicted labels of instances of x on tree tr
        :param x: data which needs to be labeled
        :param tr:
        """
        y = []

        for row in x:
            current_node = tr
            while current_node.left_child:
                if row[current_node.split_col] <= current_node.split_value:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

            label = tree_grow.majority(tr, current_node)
            y.append(label)

        self.y = y

    def return_preds(self):
        """
        Give prediction
        :return: vector of predictions
        """
        return self.y


class node:
    def __init__(self, x, dep, larger_than_bool, child=None, col=None, val=None):
        """
        Node data structure used in tree
        :param x: data matrix containing attribute values
        :param dep: depth of node
        :param larger_than_bool: decides whether it is > (True) sign or <= (False)
        :param child: child nodes of current node
        :param col: feature column on which the split is performed
        :param val: split value used for deciding left and right children
        """
        self.data = x
        self.depth = dep
        self.larger_than = larger_than_bool

        self.split_col = col
        self.split_value = val

        self.left_child = child
        self.right_child = child


# Test zooi

dataa = np.genfromtxt('pima.txt', delimiter=',', skip_header=False)

# boom laten goeien
number_of_features = (np.shape(dataa)[1] - 1)
grow_tree = tree_grow(dataa[:, :-1], dataa[:, -1], 20, 5, number_of_features)
grow_tree.give_tree()

# Test data op boom toepassen
pred = tree_pred(dataa[:, :-1], grow_tree.tree)
predictions = pred.return_preds()

arr = confusion_matrix(dataa[:, -1], predictions)
print(arr)

print("yeet")