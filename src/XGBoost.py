
import generateVector
import csv
from sklearn.metrics import accuracy_score
from docutils.parsers import null
from xgboost import XGBClassifier
import numpy as np

positiveFile="../dataset/posTrain.csv"
negativeFile="../dataset/negTrain.csv"
neutralFile="../dataset/neuTrain.csv"
XGBoost_MODEL=null

def XGBoost_model():
    global XGBoost_MODEL
    X, Y = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')
    XGBoost_MODEL = XGBClassifier()
    x_train= np.asarray(X)
    XGBoost_MODEL.fit(x_train, Y)
    print XGBoost_MODEL

def generateTestVector(testFileName):
    testVector=[]
    testLabel=[]
    f = open(testFileName, 'r')
    reader = csv.reader(f)
    for row in reader:
        try:
            a = row[2]
            z = generateVector.mapTweet(a)
            testVector.append(z)
            testLabel.append(row[1])
        except:
            None
    f.close()
    return np.asarray(testVector)

def test_XGBoost():
    global XGBoost_MODEL
    testFileName = "../dataset/test.csv"
    testvec,testLabel=generateTestVector(testFileName)
    y_pred = XGBoost_MODEL.predict(testvec)
    predictions = [round(value) for value in y_pred]
    print predictions
    # evaluate predictions
    accuracy = accuracy_score(testLabel, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

XGBoost_model()
test_XGBoost()
