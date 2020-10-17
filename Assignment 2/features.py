import extract
import numpy as np



def length(data):
    feature = np.array([[len(x[0])] for x in data[:,0]])

    data = np.append(data, feature, axis=1)
    return(feature)

if __name__ == "__main__":
    deceptiveTrain, deceptiveTest, truthfulTrain, truthfulTest = extract.extractAll()
    trainData = np.concatenate((deceptiveTrain, truthfulTrain), axis=0)
    features = np.array(length(trainData))

    #features = np.append(features,features,axis=1) #@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    features = features.transpose()
    print(features)
    trainFeatures = np.insert(trainData[:,:-1], -1, features, axis=1)
    print(trainFeatures)