import numpy as np
import os
import nltk
import re

# nltk.download('stopwords')
# nltk.download('punkt')


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
    test_data = np.array(test_data, dtype=object)
    train_data = clean_strings(train_data)
    test_data = clean_strings(test_data)
    
    return train_data, test_data


def read_fold(path, fold, legit):
    data = []
    for filename in os.listdir(path + fold):
        with open(path + fold + "/" + filename) as f:
            file_text = f.readlines()
            data.append([file_text, int(legit), filename])

    return data


def clean_strings(data):

    for row in data:
        row[0] = (row[0][0].lower())
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = nltk.stem.WordNetLemmatizer()
        filtered = [lemmatizer.lemmatize(w) for w in row[0].split() if w not in stopwords]
        tokens = nltk.tokenize.word_tokenize(" ".join(filtered))
        no_punctuation = re.sub(r'[^\w\s]', '', " ".join(tokens))
        row[0] = " ".join(no_punctuation.split())

    return data
