from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import time
from sklearn import *
import time
import sklearn
import pandas as pd
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

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

denom = 0
fold = 10 
# modify the params based on param-tuning result 
for i in range(fold):
     params = {
         'eta': 0.03333,
         'max_depth': 6,
         'objective': 'multi:softprob',
         'eval_metric': 'mlogloss',
         'num_class': 9,
         'seed': i,
         'silent': True,
         'reg_alpha': 1
     }
     # split 0.8/0.2 (train/valid)
     x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.2, random_state=i)
     watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
     # early_stop = 1000
     model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
     score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
     print(score1)
     if denom != 0:
         pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
         preds += pred
     else:
         pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
         preds = pred.copy()
     denom += 1
     # submission for each fold
     submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
     submission['ID'] = pid
     submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
     # model for each fold
     filename = 'submission_fold_'  + str(i) + '_model.sav'
     pickle.dump(model, open(filename, 'wb'))
     # accumulate (mean) submission
     preds_now = preds / denom
     submission = pd.DataFrame(preds_now, columns=['class'+str(c+1) for c in range(9)])
     submission['ID'] = pid
     submission.to_csv('submission_xgb'+str(i)+'.csv', index=False)
     t2 = time.time()
     print t1, t2, t2-t1, (t2-t1)/60

# submission: final version  
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)