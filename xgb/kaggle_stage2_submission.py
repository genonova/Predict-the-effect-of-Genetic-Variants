from sklearn import *
import sklearn
import pandas as pd
import numpy as np

sub_file = 'submission_xgb.csv'

test1 = pd.read_csv('test_variants')
filtered = pd.read_csv('stage1_solution_filtered.csv')
def whichClass(lst):
    for i, n in enumerate(lst): 
        if n==1:
            return i+1
filtered['Class'] = filtered.loc[:,'class1':'class9'].apply(whichClass, axis=1)
test1 = pd.merge(test1, filtered.loc[:,:], on='ID').fillna('')
test1['GV'] = zip(test1.Gene, test1.Variation)
test2 = pd.read_csv('stage2_test_variants.csv')
res = pd.read_csv(sub_file)
resAll = pd.merge(res, test2, on='ID')
resAll['GV'] = zip(resAll.Gene, resAll.Variation)
for r2 in range(len(resAll)):
    print '\r{:2f}%'.format(100.0*(r2+1)/len(resAll)),
    for r1 in range(len(test1)): 
        if resAll.iloc[r2,:].GV == test1.iloc[r1,:].GV:
            resAll.loc[r2, 'class1':'class9'] = test1.loc[r1, 'class1':'class9']
resAll_raw = resAll.copy()
sub = resAll.loc[:,'class1':'ID']
sub.to_csv('submission_xgb_.csv', index=False)

