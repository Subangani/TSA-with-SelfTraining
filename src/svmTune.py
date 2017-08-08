from sklearn.model_selection import GridSearchCV
from sklearn import  svm
import generateVector

positiveFile="../dataset/full_data/posTrain.csv"
negativeFile="../dataset/full_data/negTrain.csv"
neutralFile="../dataset/full_data/neuTrain.csv"
positiveTuneFile="../dataset/full_data/posTune.csv"
negativeTuneFile="../dataset/full_data/negTune.csv"
neutralTuneFile="../dataset/full_data/neuTune.csv"

X_model, Y_model = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')
X_tune, Y_tune=generateVector.loadMatrix(positiveTuneFile, neutralTuneFile, negativeTuneFile, '2', '0', '-2')

parameters = {'kernel':['linear', 'rbf'], 'C':[0.01,0.05,0.1,0.5,1.0,1.5,2.0,2.5,3.0,4.0], 'gamma':
              [0.01,0.02,0.03,0.04]}
svr = svm.SVC(class_weight={2.0: 2.4 ,-2.0: 3.3 })
svr.fit(X_model,Y_model)
grid = GridSearchCV(svr, parameters,scoring='f1_weighted', cv=5, n_jobs=-1)
tunesModel=grid.fit(X_tune, Y_tune)
print tunesModel.grid_scores_, tunesModel.best_params_,tunesModel.best_score_
