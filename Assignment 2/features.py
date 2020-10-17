import numpy as np
import extract


def add_features():
    deceptive_train, deceptive_test, truthful_train, truthful_test = extract.extract_all()

    # train_data = 640 rows, test_data = 160 rows
    train_data = np.concatenate((deceptive_train, truthful_train), axis=0)
    test_data = np.concatenate((deceptive_test, truthful_test), axis=0)
    combined_data = np.concatenate((train_data, test_data), axis=0)

    features = np.array(char_length(combined_data))

    # features = np.append(features,features,axis=1) #@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    features = features.transpose()
    data_with_features = np.insert(combined_data[:, :-1], -1, features, axis=1)

    return data_with_features


def char_length(data):
    length = np.array([[len(x[0])] for x in data[:, 0]])

    return length
