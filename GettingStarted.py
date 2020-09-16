import numpy as np

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
array2 = np.array([1, 1, 1, 1])


def impurity(arr):
    p = np.sum(arr) / len(arr)
    return p * (1 - p)


#print(impurity(array))

def bestsplit(data):
    data_sorted = data[data[:, 3].argsort()]
    income_sorted = data_sorted[:, 3]
    # print(income_sorted)
    income_len = len(income_sorted)
    income_splitpoints = (income_sorted[0:income_len-1]+income_sorted[1:income_len])/2
    # print(income_splitpoints)
    stepsize = 1 / income_len
    smallest_imp = 1
    best_income_splitpoint = 0
    for i in range(1, income_len):
        left_list = data_sorted[:, 5][:i]
        right_list = data_sorted[:, 5][i:]
        a = stepsize * i
        b = chanceof(0, left_list)
        c = 1 - b
        d = 1 - a
        e = chanceof(0, right_list)
        f = 1 - e
        imp = a * b * c + d * e * f
        print(imp)
        if imp <= smallest_imp:
            smallest_imp = imp
            best_income_splitpoint = income_splitpoints[i-1]

    return best_income_splitpoint


def chanceof(y, litty):
    count = 0
    for i in litty:
        if i == y:
            count += 1
    return count / len(litty)


#print(impurity(array))
print(bestsplit(credit_data))
