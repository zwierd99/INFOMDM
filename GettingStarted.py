import numpy as np

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
# print(credit_data)

array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
array2 = np.array([1, 1, 1, 1])


def impurity(arr):
    if len(arr) == 0:
        return 0
    else:
        p = np.sum(arr) / len(arr)
        return p * (1 - p)


def bestsplit(x, y):
    lowestImp = 1
    x_sorted = np.sort(x)
    splitpoints = (x_sorted[0:len(x) - 1] + x_sorted[1:len(x)]) / 2
    i = 0
    for point in splitpoints:
        imp1 = impurity(y[x < point])
        imp2 = impurity(y[x > point])
        totalImp = imp1 + imp2
        print(totalImp)
        if totalImp < lowestImp: #Je zou eigenlijk naar de grootste reduction moeten zoeken ipv de laagste impurity
            lowestImp = totalImp
            lowestIndex = i
        i += 1
    return (splitpoints[lowestIndex])


def split(x, i):
    return (x[0:i], x[i:])


print(bestsplit(credit_data[:, 3], credit_data[:, 5]))
