from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import robust_scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.special import boxcox1p
from scipy import stats
from scipy.stats import norm, skew
#from tqdm.autonotebook import tqdm
from tqdm import tqdm
import lightgbm as lgb
from sklearn import metrics
import gc

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, features):
        self.features =  list(set(features))
    
    def fit(self, X, *_):
        return self
    
    def transform(self, X):
        return X[self.features]

class RemoveIds(BaseEstimator, TransformerMixin):
    
    def __init__(self, allow_ids=[]):
        self.id_cols =  []
        self.allow_ids = allow_ids
    
    def fit(self, X, *_):
        self.id_cols =  [col for col in X.columns if col.endswith('_id') and col not in self.allow_ids]
        return self
    
    def transform(self, X):
        print(f'Removing id columns : {self.id_cols}')
        cols = [col for col in X.columns if col not in self.id_cols]
        return X[cols]    

class CustomMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_dict):
        self.impute_dict = impute_dict

    def fit(self, X, *_):
        return self

    def transform(self, X):
        X = X.copy()
        for col, d in self.impute_dict.items():
            for name, tran_d in d.items():
                if name == 'replace':
                     X[col] = X[col].replace(tran_d)
                elif name == 'fillna':
                    if tran_d == 'mean':
                        X[col] = X[col].fillna(X[col].mean()) 
                    elif tran_d == 'median':
                         X[col] = X[col].fillna(X[col].median()) 
                    elif tran_d == 'mode':
                         X[col] = X[col].fillna(X[col].mode()[0])
                    else:
                         X[col] = X[col].fillna(tran_d)
                elif name == 'custom_func':
                    X = tran_d(X)

        return X
        

class MissingValueImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, *_):
        return self

    def transform(self, X):
        assert type(X) == pd.DataFrame, "Please make sure input is a pandas dataframe"
        high_pec_missing = []
        for c in tqdm(X.columns, desc='Imputing Missing Values'):
            if X[c].isna().sum()/len(X[c]) >= 0.05:
                high_pec_missing.append(c)
            if X[c].dtype in ['int32', 'int64']:
                X[c].fillna(0, inplace=True)
                X[c].replace([np.inf, -np.inf], 0, inplace=True)
            elif X[c].dtype in ['float32', 'float64']:
                X[c].fillna(0.0, inplace=True)
                X[c].replace([np.inf, -np.inf], 0.0, inplace=True)
            elif X[c].dtype in ['object']:
                 X[c].fillna('UNK', inplace=True)
        print(f'Features with high perc of missing values: {high_pec_missing}')
        return X
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, catg_features):
        self.catg_features = catg_features

    def transform(self, X):
        for c in tqdm(self.catg_features, desc='Catg_Encode'):
            X[c] = X[c].astype('category')
        return X

    def fit(self, X):
        return self
        
class DataScaler(BaseEstimator, TransformerMixin):

    def __init__(self, with_centering=True, with_scaling=True):
        self.numeric_features = []
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def transform(self, X):
        for c in tqdm(self.numeric_features,desc ='Scaling'):
            X[c] = robust_scale(np.array(X[c]).reshape(-1,1), \
                                axis=0, \
                                with_centering=self.with_centering, \
                                with_scaling=self.with_scaling)   
        return X

    def fit(self, X):
        self.numeric_features = [c for c in X.columns if X[c].dtype in ['float32', 'float64', 'int32', 'int64'] ]
        return self
    
class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feats_to_log):
        self.feats_to_log = feats_to_log
        
    def transform(self, X):
        for f in tqdm(self.feats_to_log,  desc="Log+1ip -ing features"):
            X[f] = np.log1p(X[f])
        return X
    
    def fit(self, X):
        return self

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    
    def __init__(self, date_col, drop_col=True):
        self.date_col = date_col
        self.drop_col=drop_col
        self.date_features = ['year', 'day', 'month']
        
    def transform(self, X):
        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            for f in tqdm(self.date_features,  desc="Generating date feats"):
                X[f'{self.date_col}_{f}'] = X[self.date_col].dt.__getattribute__(f)
            if self.drop_col: X.drop(self.date_col, inplace=True, axis=1)
        else:
            print(f'{self.date_col} does not exist')
        return X
    
    def fit(self, X):
        return self    
    

class VarianceThresholdDF(BaseEstimator, TransformerMixin):

    def __init__(self, thres=0.0):
        self.thres = thres
        self.vt = VarianceThreshold()
        self.non_zero_var_feats = []

    def transform(self, X):
        X = X[self.non_zero_var_feats]
        return X

    def fit(self, X):
        numeric_features = [c for c in X.columns if X[c].dtype in ['float32', 'float64', 'int32', 'int64']]
        catg_features = [c for c in X.columns if c not in numeric_features]
        non_zero_var_catg_feats = [c for c in catg_features if X[c].nunique() >= 1]
        X = X[numeric_features]
        if X.isna().any().sum() != 0:
            X = MissingValueImputer().transform(X)
        self.vt.fit(X)
        features_inds = np.where(self.vt.variances_ > self.thres)[0]        
        self.non_zero_var_feats = [f for i, f in enumerate(X.columns) if i in features_inds]
        self.non_zero_var_feats = self.non_zero_var_feats + non_zero_var_catg_feats
        print('Features removed for having zero variance: ', set(X.columns) - set(self.non_zero_var_feats))
        return self
    
class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns
        self.le_enc = {}

    def transform(self, X):
        for c in tqdm(self.columns,  desc='label_enc'):
            X[c] = X[c].map(self.le_enc.get(c))
        return X

    def fit(self, X, *_):
        if self.columns is None:
            numeric_features = [c for c in X.columns if X[c].dtype in ['float32', 'float64', 'int32', 'int64']]
            self.columns = [c for c in X.columns if c not in numeric_features]
            print(f'Label encoding {len(self.columns)} columns : {self.columns}')
        self.le_enc = {c: {v: i for i, v in enumerate(
                        X[c].unique())} for c in self.columns}
        
        return self
    
    
import pickle    
class CategoricalEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.embeddings = pickle.load(open('../data/embeddings.pck', 'rb'))

    def transform(self, X):
        def get_embd(i):
            try:
                return v[0][int(i)]
            except:
                self.counter = self.counter+1
                return np.zeros(emb_sz)
            
        for k,v in tqdm(self.embeddings.items(),  desc='catg_embd'):
            c = re.sub('cat_|_emb+?','', k)
            emb_sz = len(v[0][0])
            self.counter = 0
            embds = X[c].map(get_embd) #to be used after label encoder
            print(f'{self.counter} values not found in {k}')
            embds = pd.DataFrame(np.vstack(embds), index=X.index)
            embds.columns = [f'{k}_embd_{i}' for i in range(emb_sz)]
            X = pd.concat([X,embds], axis=1)
        return X

    def fit(self, X, *_):
        return self