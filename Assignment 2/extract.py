import numpy as np
import os


def extract_all():
    deceptive_train, deceptive_test = extract(False)
    truthful_train, truthful_test = extract(True)

    return deceptive_train, deceptive_test, truthful_train, truthful_test


def extract(legit):
    test_data = []
    train_data = []
    if legit:
        path = 'data/truthful_from_Web/'
    else:
        path = 'data/deceptive_from_MTurk/'
    for fold in os.listdir(path):
        if fold == "fold5":
            test_data = test_data + read_fold(path, fold, legit)
        else:
            train_data = train_data + read_fold(path, fold, legit)
    train_data = np.array(train_data, dtype=object)

    return train_data, test_data


def read_fold(path, fold, legit):
    data = []
    for filename in os.listdir(path + fold):
        with open(path + fold + "/" + filename) as f:
            file_text = f.readlines()
            data.append([file_text, int(legit), fold])

    return data
