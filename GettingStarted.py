import numpy as np

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
array2 = np.array([1, 1, 1, 1])


def impurity(arr):
    p = np.sum(arr) / len(arr)
    return p * (1 - p)


# print(impurity(array))

def best_split(data, labels):
    """
    Computes best split value on data
    :param data:    numeric values
    :param labels:  labels (0/1) of data
    :return:    returns data value of the best possible split position
    """
    data, labels = zip(*sorted(zip(data, labels)))  # Sort both lists in the same order

    sorted_data_unique = np.array(sorted(set(data)))    # Remove duplicate values
    labels = np.array(labels)

    data_len = len(sorted_data_unique)
    split_points = (sorted_data_unique[0:data_len - 1] + sorted_data_unique[1:data_len]) / 2
    step_size = 1 / len(data)

    lowest_impurity = 1
    best_split_point = 0
    for split in split_points:  # See classification trees - 1, slide 31 for more info
        left_list = labels[data <= split]
        ratio_left = step_size * len(left_list)
        p = chance_of(0, left_list)
        left_eq = ratio_left * p * (1 - p)

        right_list = labels[data > split]
        ratio_right = 1 - ratio_left
        q = chance_of(0, right_list)
        right_eq = ratio_right * q * (1-q)

        current_impurity = left_eq + right_eq
        print(round(0.25 - current_impurity, 2))    # remove 0.25 for usage in different cases
        if current_impurity <= lowest_impurity:
            lowest_impurity = current_impurity
            best_split_point = split

    return best_split_point


def chance_of(x, litty):    # LITTY
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


print(best_split(credit_data[:, 3], credit_data[:, 5]))
