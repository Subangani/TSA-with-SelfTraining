from __future__ import division
from sklearn import svm
import numpy as np
from sklearn.cross_validation import cross_val_score


def get_Kernel_Cparameter(X,Y):
    # 5 fold cross validation
    x=np.array(X)
    y=np.array(Y)
    KERNEL_FUNCTIONS=['linear','rbf']
    C=[0.001,0.005,0.01,0.05,0.1,0.5,1.0]
    ACC=0.0
    PRE=0.0
    iter=0

    for knel in KERNEL_FUNCTIONS:
        for c in C:
            clf = svm.SVC(kernel=knel, C=c)
            scores = cross_val_score(clf, x, y, cv=5,scoring='accuracy')
            precisions=cross_val_score(clf, x, y, cv=5,scoring='precision_macro')
            if (scores.mean() > ACC and precisions.mean() > PRE):
                ACC=scores.mean()
                PRE=precisions.mean()
                KERNEL_FUNCTION=knel
                C_PARAMETER=c
            iter=iter+1
            print "iteration "+str(iter)+" : c parameter : "+str(c)+", kernel : "+str(knel)
            print("Accuracy of the model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing
            print("Precision of the model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (precisions.mean(), precisions.std() * 2))# Actual testing

    print "Optimal C : "+str(C_PARAMETER)
    print "Optimal kernel function  : "+KERNEL_FUNCTION
    print "Accurracy : "+str(ACC)
    print "Precision : "+str(PRE)
    return KERNEL_FUNCTION,C_PARAMETER