import numpy as np
import random as rd
import math


class tree_grow:
    def __init__(self, x, y):#, nmin, minleaf, nfeat):
        x = np.array(x)
        y = np.expand_dims(np.transpose(np.array(y)), axis=1)
        print(np.shape(x))
        print(np.shape(y))
        xy_combined = np.concatenate([x, y], axis=1)
        nodelist = [xy_combined]

        while nodelist:
            rd.shuffle(nodelist)
            current_node = nodelist.pop()
            current_impurity = self.impurity(current_node)

            if current_impurity > 0:
                feature_set_indices = self.draw_features(current_node[:, :-1], (np.shape(current_node)[1]-1)) # nfeat
                impurity, split_col, split_point = self.select_feature(current_node, feature_set_indices)
                impurity_reduction = current_impurity - impurity # die ook ergens meegeven??

                left_child = current_node[current_node[split_col] <= split_point]
                right_child = current_node[current_node[split_col] > split_point]

                left_child = np.delete(left_child, split_col, 1)
                right_child = np.delete(right_child, split_col, 1)

                nodelist.append(left_child)
                nodelist.append(right_child)

    def impurity(self, current_node):
        p = np.sum(current_node[:, -1]) / len(current_node[:, -1])
        return p * (1 - p)

    def draw_features(self, x, nfeat):
        nr_columns = x.shape[1]
        column_indices = rd.sample(range(0, nr_columns), nfeat)
        return x[:, column_indices]

    def select_feature(self, current_node, feature_set_indices):
        x = current_node[:, :-1]
        y = current_node[:, -1]
        lowest_impurity = math.inf
        best_col_index = -1
        best_split_point = -1
        for col in x[:, feature_set_indices]:
            split_point, impurity = self.best_split(col, y)
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
        data, labels = zip(*sorted(zip(data, labels)))  # Sort both lists in the same order

        sorted_data_unique = np.array(sorted(set(data)))  # Remove duplicate values
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
            print(round(current_impurity, 2))
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


example_arr = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
data = credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
tree_grow(data[:, :-1], data[:, -1])
