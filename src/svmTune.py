from sklearn.model_selection import GridSearchCV
from sklearn import  svm
from sklearn.model_selection import PredefinedSplit
import generateVector
from sklearn import preprocessing as pr
#import createDict

positiveFile="../dataset/full_data/positive.csv"
negativeFile="../dataset/full_data/negative.csv"
neutralFile="../dataset/full_data/neutral.csv"

X_model, Y_model = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')
X_model_scaled = pr.scale(X_model)
X_model_normalized = pr.normalize(X_model_scaled, norm='l2')  # l2 norm
X_model = X_model_normalized
X_model = X_model.tolist()

parameters = {'kernel':['linear', 'rbf'], 'C':[0.01,0.05,0.1,0.5,1.0,1.5,2.0,2.5,3.0,4.0],
              'gamma':[0.01,0.02,0.03,0.04]}
svr = svm.SVC(class_weight={2.0: 1.47 ,-2.0: 3.12})

testFold=[]
for i in range(1,len(X_model)):
    if (i%3==1) | (i%3==2):
        #for training model
        testFold.append(-1)
    else:
        #for tuning model
        testFold.append(0)

ps = PredefinedSplit(test_fold=testFold)
grid = GridSearchCV(svr, parameters,scoring='f1_weighted',n_jobs=-1,cv=ps)
tunesModel=grid.fit(X_model, Y_model)
print tunesModel.best_params_,tunesModel.best_score_
