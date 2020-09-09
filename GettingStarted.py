import numpy as np

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
array2 = np.array([1, 1, 1, 1])


def impurity(arr):
    p = np.sum(arr) / len(arr)
    return p * (1 - p)


#print(impurity(array))

def bestsplit(x, y):
    lowestImp = 1
    for i in range(len(x)):
        arrays = split(x, i)
        imp1 = impurity(arrays[0])
        imp2 = impurity(arrays[1])
        prop1 = i/len(x)
        prop2 = 1-prop1
        totalImp = prop1*imp1 + prop2*imp2
        if totalImp < lowestImp:
            lowestImp = totalImp
            lowestIndex = i
    return x[lowestIndex]


def split(x, i):
    return x[0:i], x[i:]


bestsplit(credit_data[:, 3], credit_data[:, 5])
