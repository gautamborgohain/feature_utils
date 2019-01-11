
import matplotlib.pyplot as plt
from pdpbox import pdp, info_plots
import shap

from sklearn import tree
import graphviz 


__all__ = ['plot_target', 'plot_actual', 'plot_target_categorical' ,'plot_feature_grp_targets', 'plot_cat_grp_targets','plot_partial_dependency',
          'plot_error_dist' ,'get_shap_values' , 'plot_shape_dependence', 'plot_shap_summary', 'plot_dt_clf']

####################################################
####################################################
##          Plotting functions
####################################################
####################################################

def plot_target(X, feature_name, show_percentile=False, grid_type='percentile', target='TARGET'):
    fig, axes, _ = info_plots.target_plot(df=X, 
                                           feature=feature_name, 
                                           feature_name=feature_name, 
                                           target=target, 
                                           grid_type = grid_type,
                                           show_percentile=show_percentile)
    
def plot_actual(X, model, feature_name, model_features=None):
    if not model_features:
        model_features = [c for c in X.columns if c!='TARGET' ]
    fig, axes, _ = info_plots.actual_plot(model = model, X=X[model_features], 
                                           feature=feature_name, 
                                           feature_name=feature_name, 
                                           show_percentile=True,
                                            predict_kwds={})
    
def plot_target_categorical(X, feature_name):
    X_type = pd.get_dummies(X[feature_name].astype('object')).copy()
    X_type['TARGET'] = X['TARGET']
    fig, axes, summary_df = info_plots.target_plot(
        df=X_type, feature=[c for c in X_type.columns if c != 'TARGET'], 
        feature_name=feature_name, target='TARGET')

    
def plot_feature_grp_targets(X, feat_list):
    for c in feat_list:
        try:
            plot_target(X = X, feature_name = c)
        except Exception as e:
            print(f'{c} target distribution could not be plotted because of {e}')
            
def plot_cat_grp_targets(X, feat_list):
    for c in feat_list:
        try:
            plot_target_categorical(X = X, feature_name = c)
        except Exception as e:
            print(f'{c} target distribution could not be plotted because of {e}')        
    
def plot_partial_dependency(X, model, feature_name, 
                            frac_to_plot=1.0, 
                            model_features=None, 
                            n_clusters=10, 
                            grid_range=None):
    if not model_features:
        model_features = [c for c in X.columns if c!='TARGET' ]

    pdp_iso = pdp.pdp_isolate(model = model, 
                              dataset = X, 
                              model_features = model_features, 
                              feature = feature_name,
#                               num_grid_points=10,
                              grid_range=grid_range,
                              grid_type='equal', n_jobs=8)
    fig, axes = pdp.pdp_plot(pdp_iso,
                             feature_name, 
                             plot_lines=True, 
                             cluster=True,
                             n_cluster_centers=n_clusters,
                             frac_to_plot=frac_to_plot, 
                             plot_pts_dist=True)
    

def plot_error_dist(y_test, test_preds):
    test_errors = [ (t-p)*100 for t,p in zip(y_test, test_preds)]
    test_errors = pd.Series(test_errors).map(int).value_counts(normalize=True)
    test_errors.sort_index(inplace=True)
    test_errors = test_errors.loc[-30:30]
    plot = test_errors.plot(kind='bar', fontsize=20, figsize=(20, 10),color='r' )
    print('Perc within +-1% :', sum(test_errors.loc[-1:1])/test_errors.sum())
    print('Perc within +-3% :', sum(test_errors.loc[-3:3])/test_errors.sum())
    print('Perc within +-5% :', sum(test_errors.loc[-5:5])/test_errors.sum()) 
    return plot
    
    
def get_shap_values(X, model, subsample=0.30):
    shap_test_sample = X.sample(frac=subsample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_test_sample)
    return explainer, shap_values, shap_test_sample

def plot_shape_dependence(feature, shap_values, X, color_by=None):
    return shap.dependence_plot(feature, shap_values,X , 
                     interaction_index=color_by)

def plot_shap_summary(shap_values, shap_test_sample, max_display=10):
    fig, ax = plt.subplots()
    return shap.summary_plot(shap_values, shap_test_sample, max_display=max_display)


def plot_dt_clf(X,y, max_depth=10):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
    clf.fit(X,y)
    print(f'Score {clf.score(X,y)}')
    preds = clf.predict(X)
    print(pd.Series(preds).describe())
    print(f'Feat importance {clf.feature_importances_}')
    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=max_depth,
        feature_names=X.columns,  
        class_names=[str(v) for v in y.unique()],  
        filled=True, rounded=True,  
        special_characters=True)  
    graph = graphviz.Source(dot_data)
    return graph            
            

