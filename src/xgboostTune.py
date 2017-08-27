import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV   #Performing grid search
import generateVector
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing as pr

positiveFile="../dataset/full_data/positive.csv"
negativeFile="../dataset/full_data/negative.csv"
neutralFile="../dataset/full_data/neutral.csv"

X_model, Y_model = generateVector.loadMatrix(positiveFile, neutralFile, negativeFile, '2', '0', '-2')
X_model_scaled = pr.scale(X_model)
X_model_normalized = pr.normalize(X_model_scaled, norm='l2')  # l2 norm
X_model = X_model_normalized
X_model = X_model.tolist()

testFold = []
for i in range(1, len(X_model) + 1):
    if (i % 3 == 1) | (i % 3 == 2):
        testFold.append(0)
    else:
        testFold.append(2)
#ps = PredefinedSplit(test_fold=testFold)
gkf = list(GroupKFold(n_splits=2).split(X_model, Y_model, testFold))

def param_Test1():
    global X_model,Y_model,gkf
    param_grid = {
        'max_depth': [2,4,6,8,10],
        'min_child_weight':[1,3,5,7],
        # 'gamma':[i/10.0 for i in range(0,5)],
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100]}
    xgbclf = xgb.XGBClassifier(silent=0,objective="multi:softmax",learning_rate=0.1)

    # Run Grid Search process
    gs_clf = GridSearchCV(xgbclf, param_grid,n_jobs=-1,scoring='f1_weighted',cv=gkf)
    gs_clf.fit(np.asarray(X_model), Y_model)
    print gs_clf.best_params_,gs_clf.best_score_
    print gs_clf.grid_scores_, gs_clf.best_params_, gs_clf.best_score_
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

#param_Test1()

#{'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 3} 0.767260190997

def param_test2():
    global X_model, Y_model, gkf
    param_grid = {
        'max_depth': [5,6,7],
        'min_child_weight':[2,3,4],
        # 'gamma':[i/10.0 for i in range(0,5)],
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100]}
    xgbclf = xgb.XGBClassifier(silent=0,objective="multi:softmax")
    # Run Grid Search process

    gs_clf = GridSearchCV(xgbclf, param_grid,
                          n_jobs=1,
                          scoring='f1_weighted',cv=gkf)
    gs_clf.fit(np.asarray(X_model), Y_model)
    print gs_clf.grid_scores_, gs_clf.best_params_, gs_clf.best_score_
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

#param_test2()

def paramTest2a():
    global X_model, Y_model, gkf
    param_grid = {
        #'max_depth': [5, 6, 7],
        #'learning_rate': [0.1, 0.15, 0.2, 0.3],
        #'min_child_weight':[1,3,5,7],
        # 'gamma':[i/10.0 for i in range(0,5)],
         'subsample': [i / 10.0 for i in range(6, 10)],
         'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100]}
    xgbclf = xgb.XGBClassifier(max_depth=5,min_child_weight=2,silent=0,learning_rate=0.1,objective="multi:softmax")
    gs_clf = GridSearchCV(xgbclf, param_grid,
                          n_jobs=1,
                          scoring='f1_weighted',cv=gkf)
    gs_clf.fit(np.asarray(X_model), Y_model)
    print gs_clf.grid_scores_, gs_clf.best_params_, gs_clf.best_score_
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

#paramTest2a()

def paramTest2b():
    global X_model, Y_model, gkf
    param_grid = {
        #'max_depth': [5, 6, 7],
        # 'learning_rate': [0.1, 0.15, 0.2, 0.3],
        #'min_child_weight': [1, 3, 5, 7],
         #'gamma':[i/10.0 for i in range(0,5)],
         'subsample': [i / 10.0 for i in range(6, 10)],
         'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100]}
    xgbclf = xgb.XGBClassifier(silent=0, objective="multi:softmax",learning_rate=0.1,max_depth=7,min_child_weight=7)
    # Run Grid Search process
    gs_clf = GridSearchCV(xgbclf, param_grid,
                          n_jobs=1,
                          scoring='f1_weighted',cv=gkf)
    gs_clf.fit(np.asarray(X_model), Y_model)
    print gs_clf.grid_scores_, gs_clf.best_params_, gs_clf.best_score_
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

#paramTest2b()

def paramTest3():
    global X_model, Y_model, gkf
    param_grid = {
        # 'max_depth': [5, 6, 7],
        # 'learning_rate': [0.1, 0.15, 0.2, 0.3],
        # 'min_child_weight': [1, 3, 5, 7],
         'gamma':[i/10.0 for i in range(0,5)],
        #'subsample': [i / 10.0 for i in range(6, 10)],
        #'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100]}
    xgbclf = xgb.XGBClassifier(silent=0,objective="multi:softmax", learning_rate=0.1, max_depth=7, min_child_weight=7,
                               colsample_bytree=0.9,subsample=0.9)
    # Run Grid Search process
    gs_clf = GridSearchCV(xgbclf, param_grid,
                          n_jobs=1,
                          scoring='f1_weighted',cv=gkf)
    gs_clf.fit(np.asarray(X_model), Y_model)
    print gs_clf.grid_scores_, gs_clf.best_params_, gs_clf.best_score_
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

#paramTest3()

def paramTest4a():
    global X_model, Y_model,gkf
    param_grid = {
        # 'max_depth': [5, 6, 7],
        # 'learning_rate': [0.1, 0.15, 0.2, 0.3],
        # 'min_child_weight': [1, 3, 5, 7],
        # 'gamma': [i / 10.0 for i in range(0, 5)],
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
         'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'n_estimators': [100]}
    xgbclf = xgb.XGBClassifier(silent=0, learning_rate=0.1, max_depth=7, min_child_weight=7,gamma=0.1,
                               colsample_bytree=0.8, subsample=0.6,objective="multi:softmax")
    # Run Grid Search process
    gs_clf = GridSearchCV(xgbclf, param_grid,
                          n_jobs=1,
                          scoring='f1_weighted',cv=gkf)
    gs_clf.fit(np.asarray(X_model), Y_model)
    print gs_clf.grid_scores_, gs_clf.best_params_, gs_clf.best_score_
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

paramTest4a()


