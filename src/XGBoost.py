
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
    return np.asarray(testVector),np.asarray(testLabel)

def test_XGBoost():
    global XGBoost_MODEL
    testFileName = "../dataset/test.csv"
    testvec,testLabel=generateTestVector(testFileName)
    y_pred = XGBoost_MODEL.predict(testvec)
    #print testLabel
    length=testLabel.__len__()
    testLabelInt=[None]*length
    for i in range(0,length):
        if (testLabel[i]=="neutral"):
            testLabelInt[i]=0.0
        elif (testLabel[i] == "positive"):
            testLabelInt[i] = 2.0
        elif (testLabel[i] == "negative"):
            testLabelInt[i] = -2.0
    predictions = [round(value) for value in y_pred]
    testLabelInt = map(int, testLabelInt)
    # evaluate predictions
    accuracy = accuracy_score(testLabelInt, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return predictions

def writeTest(filename,predictions):
    f = open(filename, "r")
    reader = csv.reader(f)
    fo = open(filename + ".xgboost_result", "w")
    fo.write("old,tweet,new\n")
    i=0
    for line in reader:
        tweet = line[2]
        s = line[1]
        nl = predictions[i]
        fo.write(r'"' + str(s) + r'","' + tweet + r'","' + str(nl) + r'"' + "\n")
        i+=1
    f.close()
    fo.close()
    print "labelled test dataset is stores in : " + str(filename) + ".xgboost_result"

def getAccuracyPrecision():
    try:
        testedFile="../dataset/test.csv.xgboost_result"
        f0=open(testedFile,"r")
        line = f0.readline() #pass first line
        TP=0
        TN=0
        TNeu=0
        FP=0
        FN=0
        FNeu=0
        NeuTr=0
        NegTr=0
        TrNeg=0
        NeuNeg=0
        TrNeu=0
        NegNeu=0

        readers = csv.reader(f0)
        for tweetResult in readers:
            if ((tweetResult[0]=="positive") & (str(tweetResult[2])=="2.0")):
                TP+=1
            elif ((tweetResult[0]=="negative") & (tweetResult[2]=="-2.0")):
                TN+=1
            elif ((tweetResult[0]=="neutral") & (tweetResult[2]=="0.0")):
                TNeu+=1
            elif (((tweetResult[0]=="negative") | (tweetResult[0]=="neutral")) & (tweetResult[2]=="2.0")):
                FP+=1
            elif (((tweetResult[0]=="positive")| (tweetResult[0]=="neutral")) & (tweetResult[2]=="-2.0")):
                FN+=1
            elif (((tweetResult[0]=="positive")| (tweetResult[0]=="negative")) & (tweetResult[2]=="0.0")):
                FNeu+=1

            if ((tweetResult[0]=="positive") & (tweetResult[2]=="0.0")):
                NeuTr+=1
            elif ((tweetResult[0]=="positive") & (tweetResult[2]=="-2.0")):
                NegTr+=1
            elif ((tweetResult[0]=="negative") & (tweetResult[2]=="0.0")):
                NeuNeg+=1
            elif ((tweetResult[0]=="negative") & (tweetResult[2]=="2.0")):
                TrNeg+=1
            elif ((tweetResult[0]=="neutral") & (tweetResult[2]=="2.0")):
                TrNeu+=1
            elif ((tweetResult[0]=="neutral") & (tweetResult[2]=="-2.0")):
                NegNeu+=1
        print "TP,TN,TNeu,FP,FN,FNeu,NeuTr,NegTr,NeuNeg,TrNeg,TrNeu,NegNeu"
        print TP,TN,TNeu,FP,FN,FNeu,NeuTr,NegTr,NeuNeg,TrNeg,TrNeu,NegNeu
        acc=(TP+TN+TNeu)/((TP+TN+TNeu+FP+FN+FNeu)*1.0)
        precision_pos=(TP/((FP+TP)*1.0))
        precision_neg=(TN/((FN+TN)*1.0))
        precision_neu=(TNeu/((FNeu+TNeu)*1.0))

        recall_pos=TP/((TP+NegTr+NeuTr)*1.0)
        recall_neg=TN/((TN+TrNeg+NeuNeg)*1.0)
        recall_neu=TNeu/((TNeu+TrNeu+NegNeu)*1.0)

        F_core_pos=2*(precision_pos*recall_pos)/(precision_pos+recall_pos)
        F_core_neg=2*(precision_neg*recall_neg)/(precision_neg+recall_neg)
        F_core_neu=2*(precision_neu*recall_neu)/(precision_neu+recall_neu)

        print "acc,precision_pos,precision_neg,precision_neu,recall_pos,recall_neg,recall_neu"
        print acc,precision_pos,precision_neg,precision_neu,recall_pos,recall_neg,recall_neu
        print "F_core_pos,F_core_neg,F_core_neu"
        print F_core_pos,F_core_neg,F_core_neu
    except:
        TypeError

XGBoost_model()
filename="../dataset/test.csv"
writeTest(filename,predictions=test_XGBoost())
getAccuracyPrecision()

