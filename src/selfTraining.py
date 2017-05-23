import svm
import csv

import numpy as np
from sklearn import preprocessing as pr
import warnings
warnings.filterwarnings('ignore')

unlabeledFile="../dataset/ttt.txt"
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
        f4 = open(positiveProcessedfile, "r")
        f5 = open(negativeProcessedfile, "r")
        f6 = open(neutralProcessedfile, "r")

        print i
        for j in range(i,i+10,1):
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
        #if (acc<accuracy):
            # if (sentiment == 2.0):
            #     lines = f4.readlines()
            #     lines = lines[:-1]
            #     cWriter = csv.writer(f4, delimiter=',')
            #     for line in lines:
            #         cWriter.writerow(line)
            # if (sentiment == 0.0):
            #     lines = f6.readlines()
            #     lines = lines[:-1]
            #     cWriter = csv.writer(f6, delimiter=',')
            #     for line in lines:
            #         cWriter.writerow(line)
            # if (sentiment == -2.0):
            #     lines = f5.readlines()
            #     lines = lines[:-1]
            #     cWriter = csv.writer(f5, delimiter=',')
            #     for line in lines:
            #         cWriter.writerow(line)
addToLabeledData()



