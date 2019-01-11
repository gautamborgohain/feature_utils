from ..imports.basic_imports import *


__all__ = ['get_correlated_features', 'get_feature_importance', 'get_bootstraped_feature_importance',
           'get_group_importance', 'get_groups_to_pca']

def get_correlated_features(X:pd.DataFrame, 
                            n_groups:int=8, 
                            frac:float=0.3, 
                            method:str ='pearson', 
                            kmeans_params:dict={}):
    """
    Return feature groups.
    """
    X = X.sample(frac = frac, random_state=37)
    print('generating correlation matrix')
    corr_df = X.corr(method=method)
    print('finding groups')
    km_corr = KMeans(n_clusters=n_groups, **kmeans_params).fit_predict(corr_df.fillna(0))
    km_corr = pd.DataFrame(km_corr, index = corr_df.index, columns=['cluster']).reset_index()
    km_corr['cats'] = km_corr.groupby('cluster')['index'].apply(lambda x: (x+',').cumsum().str.strip(','))
    km_corr.drop_duplicates('cluster', inplace=True, keep='last')
    km_corr['cats'] = km_corr['cats'].map(lambda x: x.split(','))
    return km_corr['cats'].tolist()

def get_feature_importance(X:pd.DataFrame,
                           y:pd.Series,
                           frac:float=0.3, 
                           classification:bool=True):
    X = X.sample(frac=frac)
    if classification:
        model = lgb.LGBMClassifier(random_state=37)
    else:
        model = lgb.LGBMRegressor(random_state=37)
    model.fit(X,y)
    feature_importance = {col:imp for col, imp in zip(X.columns, model.feature_importances_)}
    return feature_importance

def get_bootstraped_feature_importance(X:pd.DataFrame, 
                                       y:pd.Series,
                                       rounds:int=3, 
                                       classification:bool=True):
    for round in range(rounds):
        fi_ = get_feature_importance(X, y, classification=classification)
        if round == 0:
            fi = fi_
        else:
            for k, v in fi_.items():
                fi[k] = (fi[k] + v) /2
    return fi

def get_group_importance(X:pd.DataFrame,
                         y:pd.Series,
                         feature_groups:dict, 
                         feat_importance_method:str='all',
                         classification:bool=True):
    print('getting groups feature importance')
    if feat_importance_method == 'bootstrap':
        feature_importance = get_bootstraped_feature_importance(X, rounds=5, classification=classification)
    elif feat_importance_method == 'all':
        feature_importance = get_feature_importance(X, frac=1.0, classification=classification)
    group_importance = {}
    for i,grp in enumerate(feature_groups):
        for feat in grp:
            group_importance[i] = group_importance.get(i,0) + feature_importance.get(feat,0)            
    return group_importance

def get_groups_to_pca(X:pd.DataFrame,
                      y:pd.Series,
                      n_feat_groups:int=8, 
                      select_feat_grps:str='all',
                      feat_importance_method:str='all',
                      classification:bool=False):
    feat_groups = get_correlated_features(X, n_groups=n_feat_groups, kmeans_params={'random_state':37})
    if select_feat_grps == 'all':
        feat_groups_to_pca = feat_groups
    elif select_feat_grps == 'top':
        group_importance = get_group_importance(X, 
                                                y,
                                                feature_groups=feat_groups, 
                                                feat_importance_method=feat_importance_method,
                                                classification=classification)
        group_importance_sorted = sorted(group_importance.items(), key= lambda x: x[1], reverse=True)
        grp_ids_to_pca = group_importance_sorted[:4] + group_importance_sorted[-6:-2]
        feat_groups_to_pca = [feat_groups[i[0]] for i in grp_ids_to_pca] 
    return feat_groups_to_pca
