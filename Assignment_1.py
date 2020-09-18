import numpy as np
import random as rd
import math


class tree_grow:
    def __init__(self, x, y):#, nmin, minleaf, nfeat):
        # TODO: test algorithm
        # TODO: remove used feature from other elements in node list


        # Merge the data (x) and the respective labels(y) into a numpy array and add to the nodelist
        x = np.array(x)
        y = np.expand_dims(np.transpose(np.array(y)), axis=1)

        xy_combined = np.concatenate([x, y], axis=1)

        xy_node = node(xy_combined)
        nodelist = [xy_node]

        # While there are still element left in the nodelist perform this
        while nodelist:
            # Shuffle the nodelist to pick a random current_node each time
            rd.shuffle(nodelist)
            current_node = nodelist.pop()

            current_impurity = self.impurity(current_node.data)
            if current_impurity > 0:

                # Obtain the indices of the features that are considered
                feature_set_indices = self.draw_features(current_node.data[:, :-1], (np.shape(current_node.data)[1]-1)) # nfeat

                # Obtain values of best possible split from features selected above
                impurity, split_col, split_point = self.select_feature(current_node.data, feature_set_indices)
                impurity_reduction = current_impurity - impurity # die ook ergens meegeven??

                # Determine splits according to the split point
                left_child = current_node.data[current_node.data[:, split_col] <= split_point]
                right_child = current_node.data[current_node.data[:, split_col] > split_point]
                print(left_child[:, -1])
                print(right_child[:, -1])

                # delete used column from children
                left_child = np.delete(left_child, split_col, 1)
                right_child = np.delete(right_child, split_col, 1)

                # Remove used column from other still to be explored nodes
                # TODO: denk niet dat dit al goed werkt zo
                for n in nodelist:
                    np.delete(n.data, split_col, 1)

                # Transform raw data children to node children
                left_node = node(left_child)
                right_node = node(right_child)

                nodelist.append(left_node)
                nodelist.append(right_node)

                current_node.left_child = left_node
                current_node.right_child = right_node
                print()

    def impurity(self, current_node):
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
        Computes best split value on data
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
        best_split_point = 0
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
            # print(round(current_impurity, 2))
            if current_impurity <= lowest_impurity:
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


class tree_pred:
    def __init__(self, x, tr):
        x = 0


class node:
    def __init__(self, x, child=None):
        self.data = x
        self.left_child = child
        self.right_child = child


example_arr = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
data = credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
tree_grow(data[:, :-1], data[:, -1])
