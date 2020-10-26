import numpy as np
import extract
from nltk.util import ngrams
import pandas as pd

# nltk.download('wordnet')

min_uni_gram_count = 100    # 30 is wel gucci
min_bi_gram_count = 100     # 10 is wel gucci


def add_features(with_bi_grams):
    deceptive_train, deceptive_test, truthful_train, truthful_test = extract.extract_all()

    # train_data = 640 rows, test_data = 160 rows
    train_data = np.concatenate((deceptive_train, truthful_train), axis=0)
    test_data = np.concatenate((deceptive_test, truthful_test), axis=0)
    combined_data = np.concatenate((train_data, test_data), axis=0)

    uni_gram_columns = create_uni_gram_columns(combined_data)

    features = np.array(char_length(combined_data))
    features = np.append(features, uni_gram_columns, axis=1)

    if with_bi_grams:
        bi_gram_columns = create_bi_gram_columns(combined_data)
        features = np.append(features, bi_gram_columns, axis=1)

    features = features.transpose()
    print(features.shape)
    data_with_features = np.insert(combined_data[:, :-1], -1, features, axis=1)

    return data_with_features


def create_uni_gram_columns(data):
    uni_gram_dict = {}
    for row in data:
        for uni_gram in row[0].split():
            if uni_gram in uni_gram_dict:
                uni_gram_dict[uni_gram] += 1
            else:
                uni_gram_dict[uni_gram] = 1
    non_sparse_uni_gram_dict = {}
    for key in uni_gram_dict:
        if uni_gram_dict[key] > min_uni_gram_count:
            non_sparse_uni_gram_dict[key] = uni_gram_dict[key]

    filled_uni_grams = fill_uni_gram_columns(data, non_sparse_uni_gram_dict)

    return filled_uni_grams


def fill_uni_gram_columns(data, uni_gram_dict):
    harry_df = pd.DataFrame(0, index=range(data.shape[0]), columns=uni_gram_dict.keys())
    for row in range(data.shape[0]):
        for word in data[row][0].split():

            if word in uni_gram_dict:
                harry_df.at[row, word] += 1

    return harry_df.to_numpy()


def create_bi_gram_columns(data):
    bi_gram_dict = {}
    for row in data:
        for bi_gram in ngrams(row[0].split(), 2):
            if bi_gram in bi_gram_dict:
                bi_gram_dict[bi_gram] += 1
            else:
                bi_gram_dict[bi_gram] = 1
    non_sparse_bi_gram_dict = {}
    for key in bi_gram_dict:
        if bi_gram_dict[key] > min_bi_gram_count:
            non_sparse_bi_gram_dict[key] = bi_gram_dict[key]

    filled_bi_grams = fill_bi_gram_columns(data, non_sparse_bi_gram_dict)

    return filled_bi_grams


def fill_bi_gram_columns(data, bi_gram_dict):
    harry_df = pd.DataFrame(0, index=range(data.shape[0]), columns=bi_gram_dict.keys())
    for row in range(data.shape[0]):
        for bi_gram in ngrams(data[row][0].split(), 2):
            if bi_gram in bi_gram_dict:
                harry_df.at[row, bi_gram] += 1

    return harry_df.to_numpy()


def char_length(data):
    length = np.array([[len(x[0])] for x in data[:, 0]])

    return length
