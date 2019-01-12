from .imports.basic_imports import *
from .utils.transformer_utils import *
from sklearn.pipeline import Pipeline, FeatureUnion

__all__ = ['Pipeline', 'FeatureUnion', 'FeatureSelector','RemoveIds', 'CustomMissingValueImputer','MissingValueImputer','CategoricalEncoder',
          'DataScaler', 'LogTransformer', 'DateFeatureGenerator', 'VarianceThresholdDF', 'LabelEncoder', 'PcaTransformer']


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Return DataFrame with only the `features`
    """
    def __init__(self, features:list):
        self.features =  list(set(features))

    def fit(self, X:pd.DataFrame, y:Any=None):
        return self

    def transform(self, X:pd.DataFrame):
        return X[self.features]

class RemoveIds(BaseEstimator, TransformerMixin):
    """
    Remove features ending with '_id'. Allow id-columns specified in `allow_ids`
    """
    def __init__(self, allow_ids=[]):
        self.id_cols =  []
        self.allow_ids = allow_ids
    
    def fit(self, X:pd.DataFrame, y:Any=None):
        self.id_cols =  [col for col in X.columns if col.endswith('_id') and col not in self.allow_ids]
        return self
    
    def transform(self, X:pd.DataFrame):
        print(f'Removing id columns : {self.id_cols}')
        cols = [col for col in X.columns if col not in self.id_cols]
        return X[cols]    

class CustomMissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Custom Missing value imputer. Needs `impute_dict` dictionary specifying the
    imputation strategies
    
    Example:
    ----------
    def impute_area_code(X):
        #do something with X
        return X
    
    impute_dict = {
                'maintenance_fee':{'fillna': 'mode'}, #Fill na with the mode
                'floor_level':{'fillna': 'LOW'}, #fill na with string 'LOW'
                'area_code': {'custom_func': impute_area_code}, #Apply a custom function to column
                'bedrooms':{'fillna': 1.0,
                            'replace' : {0.0 : 1.0}}, # fill na with 0 and replace 0.0 with 1.0
                'price_type': {'replace': {"''": 'NEG'}, 
                                'fillna': 'NEG'},                
                'floorarea': OrderedDict([('replace', {0:np.nan}),
                                          ('fillna' , 'mode')]) # Ordered dict to specify the order 
                }
    
    ----------
    
    """
    def __init__(self, impute_dict):
        self.impute_dict = impute_dict

    def fit(self, X:pd.DataFrame, y:Any=None):
        return self

    def transform(self, X:pd.DataFrame):
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
    """
    Missing value imputer
    integer columns are are replaced with 0
    float columns are replaced with 0.0
    string/object columns are replaced with 'UNK'
    
    prints out columns where more that 5% of values were missing
    """
    
    def fit(self, X:pd.DataFrame, y:Any=None):
        return self

    def transform(self, X:pd.DataFrame):
        assert type(X) == pd.DataFrame, "Please make sure input is a pandas dataframe"
        high_pec_missing,cols_added = [],[]
        for c in tqdm(X.columns, desc='Imputing Missing Values'):
            if X[c].isna().sum()/len(X[c]) >= 0.05:
                high_pec_missing.append(c)
           
            missing_idx = X[X[c].isna()].index     
            if len(missing_idx > 0):
                miss_col = f'{c}_is_missing'
                X[miss_col] = 0
                X.loc[missing_idx][miss_col] = 1
                cols_added.append(miss_col)
                
            if X[c].dtype in ['int32', 'int64']:
                X[c].fillna(0, inplace=True)
                X[c].replace([np.inf, -np.inf], 0, inplace=True)
            elif X[c].dtype in ['float32', 'float64']:
                X[c].fillna(0.0, inplace=True)
                X[c].replace([np.inf, -np.inf], 0.0, inplace=True)
            elif X[c].dtype in ['object']:
                 X[c].fillna('UNK', inplace=True)
        print(f'Features with high perc of missing values: {high_pec_missing}')
        print(f'Missing value cols added : {cols_added}')
        return X
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, catg_features):
        self.catg_features = catg_features

    def transform(self, X:pd.DataFrame):
        for c in tqdm(self.catg_features, desc='Catg_Encode'):
            X[c] = X[c].astype('category')
        return X

    def fit(self, X:pd.DataFrame, y:Any=None):
        return self
        
class DataScaler(BaseEstimator, TransformerMixin):

    def __init__(self, method:str ='standard', columns_to_scale:Optional[list]=None):
        self.columns_to_scale = columns_to_scale
        self.scalers = {}
        self.method = method

    def transform(self, X:pd.DataFrame):
        for c, scaler in tqdm(self.scalers.items(),desc =f'{self.method} Scaling'):
            X[c] = scaler.transform(np.array(X[c]).reshape(-1,1))
        return X

    def fit(self, X:pd.DataFrame, y:Any=None):
        if self.columns_to_scale is None:
            self.columns_to_scale = [c for c in X.columns if c not in ['target', 'TARGET']]
        for c in tqdm(self.columns_to_scale, desc = f"Fitting {self.method} Scaler"):
            if X[c].dtype in ['float32', 'float64', 'int32', 'int64']:
                if self.method == 'robust':
                    self.scalers[c] = RobustScaler(with_centering=True,
                                                   with_scaling=True).fit(np.array(X[c]).reshape(-1,1))
                elif self.method == 'standard':
                    self.scalers[c] = StandardScaler(with_mean=True,with_std=True).fit(np.array(X[c]).reshape(-1,1))
        return self
    
class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feats_to_log):
        self.feats_to_log = feats_to_log
        
    def transform(self, X:pd.DataFrame):
        for f in tqdm(self.feats_to_log,  desc="Log+1 -ing features"):
            X[f] = np.log1p(X[f])
        return X
    
    def fit(self, X:pd.DataFrame,y:Any=None):
        return self

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    
    def __init__(self, date_col, drop_col=True, date_features=['year', 'day', 'month']):
        self.date_col = date_col
        self.drop_col=drop_col
        self.date_features = date_features
        
    def transform(self, X:pd.DataFrame):
        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            for f in tqdm(self.date_features,  desc="Generating date feats"):
                X[f'{self.date_col}_{f}'] = X[self.date_col].dt.__getattribute__(f)
            if self.drop_col: del X[self.date_col] 
        else:
            print(f'{self.date_col} does not exist')
        return X
    
    def fit(self, X:pd.DataFrame, y:Any=None):
        return self    
    

class VarianceThresholdDF(BaseEstimator, TransformerMixin):

    def __init__(self, thres=0.0):
        self.thres = thres
        self.vt = VarianceThreshold()
        self.non_zero_var_feats = []

    def transform(self, X:pd.DataFrame):
        X = X[self.non_zero_var_feats]
        return X

    def fit(self, X:pd.DataFrame, y:Any=None):
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

    def transform(self, X:pd.DataFrame):
        for c in tqdm(self.columns,  desc='label_enc'):
            X[c] = X[c].map(self.le_enc.get(c))
        return X

    def fit(self, X:pd.DataFrame, y:Any=None):
        if self.columns is None:
            numeric_features = [c for c in X.columns if X[c].dtype in ['float32', 'float64', 'int32', 'int64']]
            self.columns = [c for c in X.columns if c not in numeric_features]
            print(f'Label encoding {len(self.columns)} columns : {self.columns}')
        self.le_enc = {c: {v: i for i, v in enumerate(
                        X[c].unique())} for c in self.columns}
        
        return self
    
    
class PcaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 feat_groups_to_pca: List[List[str]]=[], 
                 n_feat_groups: int = None, 
                 select_feat_grps: str ='all', 
                 feat_importance_method: str = 'all',
                 classification: bool = False):
        
        self.generate_feat_groups = True if len(feat_groups_to_pca) < 1 else False
        self.feat_groups_to_pca = feat_groups_to_pca
        self.n_feat_groups = n_feat_groups
        self.feat_importance_method = feat_importance_method
        self.classification = classification
        self.select_feat_grps = select_feat_grps #['all', 'top']
        self.pca_info = []
        
    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        if self.generate_feat_groups:
            assert y is not None , "please provide y to find feature groups or else just provide the groups"
            if self.n_feat_groups is None:
                self.n_feat_groups = X.shape[1] // 10    
            self.feat_groups_to_pca = get_groups_to_pca(X,y,n_feat_groups=self.n_feat_groups,
                                                        select_feat_grps=self.select_feat_grps,
                                                        feat_importance_method = self.feat_importance_method,
                                                        classification=self.classification)
        for grp_features in tqdm(self.feat_groups_to_pca, desc="Fitting PCA"):
            pca_components = min(10, len(grp_features)//2)
            X_sub = X[grp_features] #subset the features in grou[]
            pca = PCA(n_components=pca_components, random_state=37)
            pca.fit(X_sub)
            explained_vars = pca.explained_variance_ratio_
            total_var = 0
            for i, v in enumerate(explained_vars): #get i components that explain 99.9% variance
                total_var = v+total_var
                if total_var>=0.99:
                    break
            components_selected = i + 1
            self.pca_info.append((grp_features, pca, components_selected))
        return self

    def transform(self, X):
        i = 0
        for grp in tqdm(self.pca_info, desc = "Adding PCA features"):
            grp_features, pca, components_selected = X[grp[0]], grp[1], grp[2]
            components = pca.transform(grp_features)
            components = pd.DataFrame(components[:,0:components_selected], columns=[f'pca_grp_{i}_comp_{c}' for c in range(components_selected)])
            X = pd.concat([X,components], axis=1)
            i = i+1
        return X
        