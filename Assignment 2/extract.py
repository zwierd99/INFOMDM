import numpy as np
import os
import nltk
def extract(legit):
#Deceptive = 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dataaatjesTest = []
    dataaatjesTrain = []
    if legit:
        path = 'data/truthful_from_Web/'
    else:
        path = 'data/deceptive_from_MTurk/'
    for fold in os.listdir(path):
        if fold == "fold5":
            dataaatjesTest = dataaatjesTest + readFold(path, fold, legit)
        else:
            dataaatjesTrain = dataaatjesTrain + readFold(path, fold, legit)
    dataaatjesTrain = np.array(dataaatjesTrain, dtype=object)
    return(dataaatjesTrain, dataaatjesTest)


def readFold(path, fold, legit):
    dataatjes = []
    for filename in os.listdir(path + fold):
        with open(path + fold + "/" + filename) as f:
            fileText = f.readlines()
            dataatjes.append([fileText, int(legit), fold])
    return(dataatjes)


def extractAll():
    deceptiveTrain, deceptiveTest = extract(False)
    truthfulTrain, truthfulTest = extract(True)
    return(deceptiveTrain,deceptiveTest,truthfulTrain, truthfulTest)



