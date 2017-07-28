
from docutils.parsers import null
import generateVector
import csv
import numpy as np
from sklearn import preprocessing as pr
from sklearn import svm
import tune

positiveFile="../dataset/posTrain.csv"
negativeFile="../dataset/negTrain.csv"
neutralFile="../dataset/neuTrain.csv"
MODEL=null
scaler=null
normalizer=null


# train the model
def train_SVM_model(X,Y,knel,c): # relaxation parameter
    clf=svm.SVC(kernel=knel, C=c) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    return clf

def svm_Model():
    global scaler, normalizer,MODEL
    X, Y = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')

    # features standardization
    X_scaled = pr.scale(np.array(X))
    scaler = pr.StandardScaler().fit(X)  # to use later for testing data scaler.transform(X)

    # features Normalization
    X_normalized = pr.normalize(X_scaled, norm='l2')  # l2 norm
    normalizer = pr.Normalizer().fit(X_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test

    X = X_normalized
    X = X.tolist()

    KERNEL_FUNCTION,C_PARAMETER= tune.get_Kernel_Cparameter(X, Y)

    print "Training model with optimized parameters"
    MODEL = train_SVM_model(X, Y, KERNEL_FUNCTION, C_PARAMETER)
    print "Training done !"

#predict a tweet using the model
def predict(tweet,model): # test a tweet against a built model
    global scaler, normalizer
    z = generateVector.mapTweet(tweet)  # mapping
    z_scaled = scaler.transform(z)
    z = normalizer.transform([z_scaled])
    z = z[0].tolist()
    return model.predict([z]).tolist()[0]  # transform nympy array to list

# write labelled  test dataset
def writeTest(filename,model): # function to load test file in the csv format : sentiment,tweet
    f = open(filename, "r")
    reader = csv.reader(f)
    fo=open(filename+".svm_result","w")
    fo.write("old,tweet,new\n")
    for line in reader:
        tweet = line[2]
        s = line[1]
        nl=predict(tweet,model)
        fo.write(r'"'+str(s)+r'","'+tweet+r'","'+str(nl)+r'"'+"\n")
    f.close()
    fo.close()
    print "labelled test dataset is stores in : "+str(filename)+".svm_result"

def test(model):
    print "Loading test data..."
    writeTest('../dataset/test.csv', model)
    getAccuracyPrecision()

def getAccuracyPrecision():
    testedFile="../dataset/test.csv.svm_result"
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

svm_Model()
test(MODEL)