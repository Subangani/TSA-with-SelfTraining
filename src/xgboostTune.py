
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import generateVector
from xgboost import XGBClassifier

positiveFile="../dataset/full_data/posTrain.csv"
negativeFile="../dataset/full_data/negTrain.csv"
neutralFile="../dataset/full_data/neuTrain.csv"
positiveTuneFile="../dataset/full_data/posTune.csv"
negativeTuneFile="../dataset/full_data/negTune.csv"
neutralTuneFile="../dataset/full_data/neuTune.csv"

X_model, Y_model = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')
X_tune, Y_tune=generateVector.loadMatrix(positiveTuneFile, neutralTuneFile, negativeTuneFile, '2', '0', '-2')

def param_Test1():
    global X_model,Y_model,X_tune,Y_tune
    cv_params = {'max_depth': [3,5,7,9], 'min_child_weight': [1,3,5]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 50, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'objective': 'multi:softmax','silent':0}
    optimized_GBM = GridSearchCV(XGBClassifier(**ind_params).fit(np.asarray(X_model),Y_model),param_grid=cv_params,scoring = 'accuracy', cv = 5, n_jobs = -1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    tunedModel=optimized_GBM.fit(np.asarray(X_tune), Y_tune)
    print tunedModel
    print tunedModel.grid_scores_,tunedModel.best_params_,tunedModel.best_score_

param_Test1()

# def param_test2():
#
#     cv_params = {'max_depth':range(3, 10, 2),
#                  'min_child_weight':range(1, 6, 2)}
#     ind_params = {'learning_rate': 0.1, 'n_estimators': 140, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
#                   'objective': 'multi:softmax', 'silent': 0}
#     optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
#                                  cv_params,
#                                  scoring='accuracy', cv=5, n_jobs=-1)
#     # Optimize for accuracy since that is the metric used in the Adult Data Set notation
#
#     optimized_GBM.fit(np.asarray(X_model), Y_model)
#     print optimized_GBM.grid_scores_


# param_test2 = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[4,5,6]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier( silent=0,learning_rate=0.1,
#                         n_estimators=140, subsample=0.8, colsample_bytree=0.8,
#                         objective= 'multi:softmax', nthread=4,seed=0),
#                 param_grid = param_test2, scoring='accuracy',n_jobs=-1, cv=5)
# gsearch2.fit(np.asarray(X_model), Y_model)
# print gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

def paramTest2a():
    cv_params = {'max_depth': [2,3,4], 'min_child_weight': [4,5,6]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 140, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                     'objective': 'multi:softmax','silent':0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                    cv_params,
                                     scoring = 'accuracy', cv = 5, n_jobs = -1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_,optimized_GBM.best_params_,optimized_GBM.best_score_

def paramTest2b():
    cv_params = { 'min_child_weight': [5,6,7]}
    ind_params = {'learning_rate': 0.1,'max_depth':3, 'n_estimators': 140, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                     'objective': 'multi:softmax','silent':0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                    cv_params,
                                     scoring = 'accuracy', cv = 5, n_jobs = -1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_,optimized_GBM.best_params_,optimized_GBM.best_score_



def paramTest3():
    cv_params = { 'gamma':[i/10.0 for i in range(0,5)]}
    ind_params = {'learning_rate': 0.1,'max_depth':3,'min_child_weight':6, 'n_estimators': 140, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                     'objective': 'multi:softmax','silent':0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                    cv_params,
                                     scoring = 'accuracy', cv = 5, n_jobs = -1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_,optimized_GBM.best_params_,optimized_GBM.best_score_



def paramTest4a():
    cv_params = { 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]}
    ind_params = {'learning_rate': 0.1,'max_depth':3,'min_child_weight':6, 'n_estimators': 140, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                     'objective': 'multi:softmax','silent':0,'gamma':0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                    cv_params,
                                     scoring = 'accuracy', cv = 5, n_jobs = -1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_,optimized_GBM.best_params_,optimized_GBM.best_score_

def paramTest4b():
    cv_params = {'subsample':[i / 100.0 for i in range(60, 80, 5)],
    'colsample_bytree':[i / 100.0 for i in range(50, 70, 5)]}
    ind_params = {'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 140, 'seed': 0,
                  'objective': 'multi:softmax', 'silent': 0, 'gamma': 0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                 cv_params,
                                 scoring='accuracy', cv=5, n_jobs=-1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_, optimized_GBM.best_params_, optimized_GBM.best_score_




def paramTest5a():
    cv_params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
    ind_params = {'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators':140, 'seed': 0,
                  'subsample': 0.7, 'colsample_bytree': 0.6,
                  'objective': 'multi:softmax', 'silent': 0, 'gamma': 0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                 cv_params,
                                 scoring='accuracy', cv=5, n_jobs=-1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_, optimized_GBM.best_params_, optimized_GBM.best_score_




def paramTest5b():
    cv_params = {'reg_alpha':[0.000005, 0.00001, 0.000015, 0.00002, 0.000025]}
    ind_params = {'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 140, 'seed': 0,
                  'subsample': 0.7, 'colsample_bytree': 0.6,
                  'objective': 'multi:softmax', 'silent': 0, 'gamma': 0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                 cv_params,
                                 scoring='accuracy', cv=5, n_jobs=-1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_, optimized_GBM.best_params_, optimized_GBM.best_score_

def paramTest6():
    cv_params = {'n_estimators':[0,50,100,500]}
    ind_params = {'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 6,  'seed': 0,
                  'subsample': 0.7, 'colsample_bytree': 0.6,'reg_alpha':1e-05,
                  'objective': 'multi:softmax', 'silent': 0, 'gamma': 0}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                 cv_params,
                                 scoring='accuracy', cv=5, n_jobs=-1)
    # Optimize for accuracy since that is the metric used in the Adult Data Set notation

    optimized_GBM.fit(np.asarray(X_model), Y_model)
    print optimized_GBM.grid_scores_, optimized_GBM.best_params_, optimized_GBM.best_score_


