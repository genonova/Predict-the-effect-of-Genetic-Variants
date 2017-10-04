from sklearn import *
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV 
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pickle
import time
# run from data engineering or load the existed feature data
from feature import train, test, y
# Load 
# with open('./y', 'r') as f:
#     y = pickle.load(f)
# with open('./test', 'r') as f:
#     test = pickle.load(f)
# with open('./train', 'r') as f:
#     train = pickle.load(f)
# print y.shape, train.shape, test.shape

# optimal n_estimators
def modelfit(algo, dFeature, dLabel, 
             useTrainCV=True, cv_fold = 5,
             early_stopping_rounds = 50,
             verbose_eval = 50):
    if useTrainCV:
        xgb_param = algo.get_xgb_params()
        xgb_param['num_class'] = np.unique(dLabel).shape[0]
        xgtrain = xgb.DMatrix(dFeature, label=dLabel)
        cvRes = xgb.cv(xgb_param, xgtrain, 
                       num_boost_round = algo.get_xgb_params()['n_estimators'],
                       nfold = cv_fold, 
                       metrics='mlogloss',
                       early_stopping_rounds=early_stopping_rounds,
                       show_stdv = False,
                       verbose_eval = verbose_eval)
        algo.set_params(n_estimators=cvRes.shape[0])
    return cvRes, cvRes.shape[0]

xgb_model = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=4,
    scale_pos_weight=1,
    seed=826)


_, n = modelfit(xgb_model, train, y)
print n
xgb_model.set_params(n_estimators = n)

# max_depth & min_child_weight $ gamma
# subsample & colsample_bytree
# reg_alpha (reg_lambda)
param_test1 = {
 'max_depth':range(4,10,2),
 'min_child_weight':range(1,6,2),
 'gamma':[i/10.0 for i in range(0,5,2)]
 # 'subsample':[i/10.0 for i in range(7,11,1)],
 # 'colsample_bytree':[i/10.0 for i in range(7,11,1)]
 # 'reg_alpha':[1e-5, 1e-3, 1e-2, 0.1, 1, 10]
}
gsearch1 = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_test1, 
                        scoring='neg_log_loss',
                        n_jobs=4, 
                        iid=False, 
                        cv=5)
gsearch1.fit(train, y)
print gsearch1.grid_scores_ 
print gsearch1.best_params_ 
print gsearch1.best_score_

xgb_model.set_params(max_depth = gsearch1.best_params_['max_depth'],
                     min_child_weight = gsearch1.best_params_['min_child_weight'],
                     gamma = gsearch1.best_params_['gamma'])

# xgb_model.set_params(n_estimators = 1000)
# _, n = modelfit(xgb_model, train, y)
# print n
# xgb_model.set_params(n_estimators = n)

pickle.dump(xgb_model, open('xgb1', 'wb'))