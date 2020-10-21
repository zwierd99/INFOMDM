import numpy as np
import extract
import nltk
from nltk.util import ngrams
minUnigramCount = 100
minBigramCount = 20

def add_features():
    deceptive_train, deceptive_test, truthful_train, truthful_test = extract.extract_all()

    # train_data = 640 rows, test_data = 160 rows
    train_data = np.concatenate((deceptive_train, truthful_train), axis=0)
    test_data = np.concatenate((deceptive_test, truthful_test), axis=0)
    combined_data = np.concatenate((train_data, test_data), axis=0)
    document_matrix = create_document_matrix(combined_data)

    features = np.array(char_length(combined_data))
    #features = np.append(features, document_matrix, axis=1)
    # features = np.append(features,features,axis=1) #@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    features = features.transpose()
    data_with_features = np.insert(combined_data[:, :-1], -1, features, axis=1)

    return data_with_features



def create_document_matrix(data):
    unigramDict = {}
    bigramDict = {}
    for row in data:
        for bigram in ngrams(row[0][0].split(), 2):
            if bigram in bigramDict:
                bigramDict[bigram] += 1
            else:
                bigramDict[bigram] = 1
        for unigram in row[0][0].split():
            if unigram in unigramDict:
                unigramDict[unigram] += 1
            else:
                unigramDict[unigram] = 1
    #print(corpus)
    print(len(bigramDict))
    nonSparseUnigramDict = {}
    for key in unigramDict:
        if unigramDict[key] > minUnigramCount:
            nonSparseUnigramDict[key] = unigramDict[key]
    nonSparseBigramDict = {}
    for key in bigramDict:
        if bigramDict[key] > minBigramCount:
            nonSparseBigramDict[key] = bigramDict[key]

    print(nonSparseBigramDict)
    print(len(nonSparseBigramDict))
    #return corpus


def char_length(data):
    length = np.array([[len(x[0])] for x in data[:, 0]])

    return length

add_features()