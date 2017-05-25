import svm
import preprocess

import csv

import numpy as np
from sklearn import preprocessing as pr
import warnings
warnings.filterwarnings('ignore')

unlabeledFile="../dataset/unlabeled.txt"
positiveProcessedfile="../dataset/positiveProcessed.txt"
negativeProcessedfile="../dataset/negativeProcessed.txt"
neutralProcessedfile = "../dataset/neutralProcessed.txt"
accuracy=svm.accuracy
print accuracy

def addToLabeledData():
    model = svm.MODEL
    for i in range(1,300,10):
        f0 = open(unlabeledFile, "r")
        f1 = open(positiveProcessedfile, "a")
        f2 = open(negativeProcessedfile, "a")
        f3 = open(neutralProcessedfile, 'a')

        print i
        for j in range(i,i+50,1):
            print j
            tweet = f0.readline()
            sentiment=svm.predict(tweet,model)
            print sentiment
            if (sentiment==float(2.0)):
                f1.write(tweet)
            if (sentiment==float(0.0)):
                f3.write(tweet)
            if (sentiment==float(-2.0)):
                f2.write(tweet)
        f1.close()
        f2.close()
        f3.close()
        preprocess.ngramgeneration(positiveProcessedfile, svm.positiveUnigram, svm.positiveBigram, svm.positiveTrigram)
        preprocess.ngramgeneration(negativeProcessedfile, svm.negativeUnigram, svm.negativeBigram, svm.negativeTrigram)
        preprocess.ngramgeneration(neutralProcessedfile, svm.neutralUnigram, svm.neutralBigram, svm.neutralTrigram)
        X, Y = svm.loadMatrix(positiveProcessedfile, neutralProcessedfile,negativeProcessedfile, '2', '-2', '0')
        X_scaled = pr.scale(np.array(X))

        # features Normalization
        X_normalized = pr.normalize(X_scaled, norm='l2')  # l2 norm
        X = X_normalized
        X = X.tolist()
        KERNEL_FUNCTION = 'linear'
        C_PARAMETER = 1.0

        print "Training model with optimized parameters"
        model = svm.trainModel(X, Y, KERNEL_FUNCTION, C_PARAMETER)
        print "Training done !"
        svm.writeTest('../dataset/test.csv', model)
        acc,pre=svm.getAccuracyPrecision()
        print "*********"
        print acc,pre,accuracy

addToLabeledData()



