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

def whichClass(lst):
    for i, n in enumerate(lst): 
        if n==1:
            return i+1

# transform word 2 vec and sum up for each paper; calculate average vector (1*200) for each. 
def pp2vec(pp):
    vec = np.zeros(200)
    n_words = 0
    list_of_words = [i.lower() for i in wordpunct_tokenize(pp) if i.lower() not in stop_words]
    for word in list_of_words:
        try:
            vec += model[word]
            n_words += 1
        except:
            pass
    return vec / n_words

def pps2vec(pps):
    vecs = pps.Text.map(pp2vec)
    vecs = vecs.apply(pd.Series)
    vecs = vecs.rename(columns = lambda x : 'Vec_' + str(x+1))
    pps = pd.concat([pps[:], vecs[:]], axis=1)
    return pps

# calculate the ratio of words in the selected word2vec dictionary.    
def pp2vecRatio(pp):
    vec = np.zeros(200)
    n_words = 0
    list_of_words = [i.lower() for i in wordpunct_tokenize(pp) if i.lower() not in stop_words]
    for word in list_of_words:
        try:
            vec += model[word]
            n_words += 1
        except:
            pass
    return 1.0*n_words/len(list_of_words)

def df2vec(df, fname):
    t1 = time.time()
    df_new = pps2vec(df)
    # df_new.to_csv(fname, index=False)
    df_new['Ratio'] = df_new.Text.map(pp2vecRatio)
    print 'finish '+fname
    t2 = time.time()
    print t2-t1, t2/60 - t1/60
    return df_new

# def addRation(fname):
#     df = pd.read_csv(fname)
#     df['Ratio'] = df.Text.map(pp2vecRatio)
#     df.to_csv(fname, index=False)
#     print 'finish '+fname
#     return df



# load dictionary
word2vec_path='./PubMed-and-PMC-w2v.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)  # C binary format

# customize stop words list
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '-']) 

# read text data: 
# train-> training data (without text)
# test1-> test data (stage 1)
# filtered-> part of labelled test data (stage 1); take them as training data for (stage 2) training.
train = pd.read_csv('training_variants')
test1 = pd.read_csv('test_variants')
filtered = pd.read_csv('stage1_solution_filtered.csv')
test2 = pd.read_csv('stage2_test_variants.csv')

# trainx-> training text data; 
# test1x-> testing text data (stage 1)
# test2x-> testing text data (stage 2)
trainx = pd.read_csv('training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test1x = pd.read_csv('test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test2x = pd.read_csv('stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
trainx = df2vec(trainx, 'training_text_vec.csv')
test1x = df2vec(test1x, 'test_text_vec.csv')
test2x = df2vec(test2x, 'stage2_test_text_vec.csv')
# addRation('stage2_test_text_vec.csv')
# addRation('training_text_vec.csv')
# addRation('test_text_vec.csv')


# train: combine train and test1 (labelled data)
filtered['Class'] = filtered.loc[:,'class1':'class9'].apply(whichClass, axis=1)
test1 = pd.merge(test1, filtered.loc[:,['ID', 'Class']], on='ID').fillna('')
test1 = pd.merge(test1, test1x, how='left', on='ID').fillna('')
train = pd.merge(train, trainx, how='left', on='ID').fillna('')
train = train.append(test1)
train.reset_index(drop=True, inplace=True)
print train.shape
y = train['Class'].values
train = train.drop(['Class'], axis=1)
# test (for stage 2)
test = pd.merge(test2, test2x, how='left', on='ID').fillna('')
print test.shape
pid = test['ID'].values
# concat train + test
df_all = pd.concat((train, test), axis=0, ignore_index=True)

# unique gene+variation list
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in [r['Variation']] if w in r['Text']]), axis=1)
gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
i_ = 0
for gen_var_lst_itm in gen_var_lst: # word-level freq
    if i_ % 100 == 0: print(i_),
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1
for i in range(20): # char-level freq (first 20 chars; median=18, max=55)
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')


# Bioinformatics Code + alphabet feature engineering 
"""
## Bioinformatics Code + alphabet feature engineering 
from: https://github.com/ddofer/ProFET/blob/master/ProFET/feat_extract/AAlphabets.py
ProFET: Feature engineering captures high-level protein functions.
Ofer D, Linial M.
Bioinformatics. 2015 Nov 1;31(21):3429-36. doi: 10.1093/bioinformatics/btv345.
PMID: 26130574
"""
def TransDict_from_list(groups):
    '''
    Given a list of letter groups, returns a dict mapping each group to a
    single letter from the group - for use in translation.
    '''
    transDict = dict()
    result = {}
    for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            result[c] = str(g_members[0]) #K:V map, use group's first letter as represent.
    return result
ofer8 = TransDict_from_list(["C", "G", "P", "FYW", "AVILM", "RKH", "DE", "STNQ"])
sdm12 = TransDict_from_list(["A", "D", "KER", "N",  "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"] )
pc5 = {"I": "A", # Aliphatic
       "V": "A", 
       "L": "A",
       "F": "R", # Aromatic
       "Y": "R", 
       "W": "R", 
       "H": "R",
       "K": "C", # Charged
       "R": "C", 
       "D": "C", 
       "E": "C",
       "G": "T", # Tiny
       "A": "T", 
       "C": "T", 
       "S": "T",
       "T": "D", # Diverse
       "M": "D", 
       "Q": "D", 
       "N": "D",
       "P": "D"}
# new features
AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'
df_all["simple_variation_pattern"] = df_all.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)
# Get location in gene / first number , from first word (otherwise numbers appear later)
df_all['location_number'] = df_all.Variation.str.extract('(\d+)')
df_all['variant_letter_first'] = df_all.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else np.NaN,axis=1)
df_all['variant_letter_last'] = df_all.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else np.NaN ,axis=1)
# Replace letters with NaNs for cases that don't match our pattern. (Need to check if this actually improves results!)"
df_all.loc[df_all.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = np.NaN
# encode the reduced alphabet as OHE features; in peptidomics this gives highly generizable features."
df_all['AAGroup_ofer8_letter_first'] = df_all["variant_letter_first"].map(ofer8)
df_all['AAGroup_ofer8_letter_last'] = df_all["variant_letter_last"].map(ofer8)
df_all['AAGroup_ofer8_equiv'] = df_all['AAGroup_ofer8_letter_first'] == df_all['AAGroup_ofer8_letter_last']
df_all['AAGroup_m12_equiv'] = df_all['variant_letter_last'].map(sdm12) == df_all['variant_letter_first'].map(sdm12)
df_all['AAGroup_p5_equiv'] = df_all['variant_letter_last'].map(pc5) == df_all['variant_letter_first'].map(pc5)

# labelencoder for gene&var names; len&words for names and paper 
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))          

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

# concat features
class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([
            	('Gene', cust_txt_col('Gene')), 
            	('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), 
            	('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([
            	('Variation', cust_txt_col('Variation')),
            	('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), 
            	('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi3', pipeline.Pipeline([
            	('Text', cust_txt_col('Text')), 
            	('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), 
            	('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)
y = y - 1 #fix for zero bound array

