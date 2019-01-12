
from math import sqrt
from copy import deepcopy
from dateutil.relativedelta import relativedelta
import scipy.stats as ss
import math

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
from sklearn import metrics
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import List
from collections import Counter

import lightgbm as lgb

__all__ = ['train_test_lightgbm', 'kfold_lightgbm', 'time_series_lightgbm', 'display_importances', 'MixedInputModel']

def calc_rmse(y_true, y_pred):
    if type(y_true) in ['list', pd.Series]:
        return sqrt(metrics.mean_squared_error(y_true, y_pred))
    else:
        return sqrt(((y_true-y_pred)**2)/2)

def train_test_lightgbm(df, lgbm_params={}, classification=True, split=0.33, early_stopping_rounds=300, eval_metric=None,score_submission=False,test_df=None, random_state=42):
    """
    Returns model, X_train, X_test, train_preds, y_train, y_test , test_preds
    
    
    if score_submission is True, returns:
    model, X_train, X_test, train_preds, y_train, y_test , test_preds, submission_preds
    """
    y = df['TARGET']
    X = df.drop('TARGET', axis=1)
    del df
    gc.collect()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=random_state)
    print(len(X_train), len(X_test))
    print('Training Model ... ')
    if classification:
        eval_metric='multi_logloss,multi_error' if eval_metric is None else eval_metric
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train, \
                    early_stopping_rounds=early_stopping_rounds, \
                    eval_set=[(X_train, y_train),(X_test,y_test)],\
                    eval_metric=eval_metric,\
                    verbose=100)
    else:
        eval_metric='poisson,rmse,mae' if eval_metric is None else eval_metric
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train, \
                    early_stopping_rounds=early_stopping_rounds, \
                    eval_set=[(X_train, y_train),(X_test,y_test)],\
                    eval_metric=eval_metric,\
                    verbose=100)
    
    print('Evaluating ... ')
    train_preds = model.predict(X_train, num_iteration=model.best_iteration_)
    test_preds = model.predict(X_test, num_iteration=model.best_iteration_)
    
    if classification:
        train_score = metrics.accuracy_score(y_train, train_preds)
        test_score = metrics.accuracy_score(y_test, test_preds)
        print('Train, Test Accuracy : ', train_score, test_score)
        print('Test F1 : ', metrics.f1_score(y_test, test_preds,average='macro'))
        print(metrics.classification_report(y_test,test_preds))
    
    else:
        train_score = calc_rmse(y_train, train_preds)
        test_score = calc_rmse(y_test, test_preds)
        print('Train, Test RMSE : ', train_score, test_score)
        print('Test R2 : ', metrics.r2_score(y_test, test_preds))
        print('mean_absolute_error : ', metrics.mean_absolute_error(y_test, test_preds))
        print('RMSE : ', sqrt(metrics.mean_squared_error(y_test, test_preds)))
    if score_submission:
        submission_preds = model.predict(test_df[X_train.columns], num_iteration=model.best_iteration_)  
        return model, X_train, X_test, train_preds, y_train, y_test , test_preds, submission_preds

    return model, X_train, X_test, train_preds, y_train, y_test , test_preds
    

def kfold_lightgbm(df, lgbm_params={}, num_folds=4, classification=True, stratified = False, feats_sub=None, eval_metric=None,score_submission=False,test_df=None,random_state=1001):
    """
    Returns feature_importance_df
    
    if score_submission = True, returns feature_importance_df,test_df
    """
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    if score_submission:
        submission_preds = np.zeros(test_df.shape[0])
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified and classification:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=random_state)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=random_state)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    
    feature_importance_df = pd.DataFrame()
    if feats_sub:
        feats = list(set(feats_sub) & set(train_df.columns)) + ['TARGET']
    else:
        feats = [f for f in train_df.columns if f not in ['TARGET']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        if classification:
            eval_metric = 'auc' if eval_metric is None else eval_metric
            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= eval_metric, 
                      verbose= 100, 
                      early_stopping_rounds= 200)
            oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
            if score_submission:
                submission_preds += model.predict_proba(test_df[feats], num_iteration=model.best_iteration_)[:, 1] / folds.n_splits
        else:
            eval_metric = 'rmse,mae,poisson' if eval_metric is None else eval_metric
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= eval_metric, 
                      verbose= 100, 
                      early_stopping_rounds= 300)
            oof_preds[valid_idx] = model.predict(valid_x, num_iteration=model.best_iteration_)
            if score_submission:
                submission_preds += model.predict(test_df[feats], num_iteration=model.best_iteration_) / folds.n_splits
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if classification:
            print('Fold %2d AUC : %.6f' % (n_fold + 1, metrics.roc_auc_score(valid_y, oof_preds[valid_idx])))
        else:
            print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(metrics.mean_squared_error(valid_y, oof_preds[valid_idx]))))
            
        if score_submission:
            test_df['TARGET'] = submission_preds
            
            
        del model, train_x, train_y, valid_x, valid_y
        gc.collect()
    
    if classification:
        print('Full AUC score %.6f' % metrics.roc_auc_score(train_df['TARGET'], oof_preds))
    else:
        print('Full RMSE score %.6f' % np.sqrt(metrics.mean_squared_error(train_df['TARGET'], oof_preds)))
        
    if score_submission:
        return feature_importance_df, test_df
    else:
        return feature_importance_df


def time_series_lightgbm(df, lgbm_params={}, classification=True):

    assert 'calendar_week' in df.columns, 'need calendar_week to split'
    df['calendar_week'] = pd.to_datetime(df['calendar_week'])
    start_week = df['calendar_week'].min()
    max_week = df['calendar_week'].max()
    diff = max_week - start_week
    diff = diff.days
    month_cursor = start_week + relativedelta(months=1)
    for i in range( round(diff/30) ) :
        one_month = month_cursor + relativedelta(months=1)
        y_train = df[ (df.calendar_week <= month_cursor) ]['TARGET']
        X_train = df[ (df.calendar_week <= month_cursor) ]
        X_train.drop(['TARGET','calendar_week'], axis=1, inplace=True)
        y_test = df[ (df.calendar_week > month_cursor) & (df.calendar_week <= one_month) ]['TARGET']
        X_test = df[ (df.calendar_week > month_cursor) & (df.calendar_week <= one_month) ]
        X_test.drop(['TARGET','calendar_week'], axis=1, inplace=True)
    
        print(len(X_train), len(X_test))

        print('Training Model ... ')
        if classification:
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X_train, y_train, \
                        early_stopping_rounds=100, \
                        eval_set=(X_test,y_test),\
                        eval_metric='auc,accuracy',\
                        verbose=100)
        else:
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X_train, y_train, \
                        early_stopping_rounds=100, \
                        eval_set=(X_test,y_test),\
                        eval_metric='poisson,rmse,mae',\
                        verbose=100)

        print('Evaluating ... ')
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        if classification:
            train_score = metrics.accuracy_score(y_train, train_preds)
            test_score = metrics.accuracy_score(y_test, test_preds)
            print(f'Train from {start_week} to {month_cursor}')
            print(f'Testing from {month_cursor} to {one_month}')            
            print('Train, Test Accuracy : ', train_score, test_score)
            print('Test F1 : ', metrics.f1_score(y_test, test_preds,average='macro'))
            print(metrics.classification_report(y_test,test_preds))

        else:
            train_score = calc_rmse(y_train, train_preds)
            test_score = calc_rmse(y_test, test_preds)
            print(f'Train from {start_week} to {month_cursor}')
            print(f'Testing from {month_cursor} to {one_month}')             
            print('Train, Test RMSE : ', train_score, test_score)
            print('Test R2 : ', metrics.r2_score(y_test, test_preds))
            print('mean_absolute_error : ', metrics.mean_absolute_error(y_test, test_preds))
            print('RMSLE : ', sqrt(metrics.mean_squared_log_error(y_test, test_preds)))
            print('RMSE : ', sqrt(metrics.mean_squared_error(y_test, test_preds)))
        
        month_cursor = month_cursor + relativedelta(months=1)
    
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",\
                                                                                                   ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    return plt


# import keras
# from keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Flatten, concatenate
# from keras.models import Model

class MixedInputModel:
    def __init__(self):
        import keras
        from keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Flatten, concatenate
        from keras.models import Model

    def __call__(self, cont_sz, emb_szs):
        cont_input = Input(shape=(cont_sz,), dtype='float32', name='cont_input')
        def cat_input(emb):
            return Input(shape=(1,),  name = f'cat_{emb[0]}_input')
        
        def emb_init(cat_input, emb):
            cat_emb  = Embedding(emb[1], emb[2],\
                                embeddings_initializer='uniform',\
                                name=f'cat_{emb[0]}_emb')(cat_input)
            cat_emb = Flatten()(cat_emb)
            return cat_emb
        
        cat_inputs = [cat_input(emb) for emb in emb_szs]
        x = concatenate([cont_input] + [emb_init(cat_input, emb) for cat_input, emb in zip(cat_inputs,emb_szs)])
        x = BatchNormalization()(x)
        x = Dense(1000, activation='relu', name ='fc_1')(x)
        x = BatchNormalization()(x)
        x = Dense(500, activation='relu', name ='fc_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation='sigmoid', name = 'output')(x)
        model = Model(inputs = [cont_input] + cat_inputs, outputs = x)
        opt = keras.optimizers.Adam(lr=0.01)
        model.compile(optimizer = opt, loss='mean_squared_error', metrics=['mean_squared_error', 'mae', 'poisson'])
        return model
