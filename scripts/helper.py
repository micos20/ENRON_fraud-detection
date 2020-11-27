#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

def crt_plot(data, features, type='box', figsize=(20,15), shape=(1,1), log=[], bins=10, sort=True, color='orange', save=False):
    '''
    Creates multiple boxplots from dataframe and organizes plots in 2D matrix.
    ---
    data:        pandas data frame
    features:    list of column names of data frame to be plotted
    figsize:     tuple of width x hight
    shape:       tuple of rows x columns in fig
    log:         list of features with logarithmic y axis scale (only necessary when your data contains outlyers)
    sort:        boolean, if True features will be sorted
    color:       color string, see matplotlib docs for valid colors
    save:        filename as string or False
    
    Returns None
    '''
    rows = shape[0]
    cols = shape[1]
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    if sort == True:
        plot_iter = iter( sorted(features) )
    else:
        plot_iter = iter( features )
    for row in range(rows):
        for col in range(cols):
            column = next(plot_iter, 'STOP')
            if column == 'STOP':
                continue
            else:
                sns.boxplot(data=data[column], ax=ax[row,col], color=color)
                ax[row,col].set_title(column)
                if column in log:
                    ax[row,col].set_yscale('log')
    plt.tight_layout()
    if save is not False:
        plt.savefig(save, dpi='figure')
    plt.show()
    return None

# Correlates feature combinations of data frame columns with selected feature
def correlate(frame, corr, feature_list=None):
    correlations = []
    if feature_list == None:
        feature_list = frame.columns
    for feature in feature_list:
        for divisor in feature_list:
            if feature == divisor:
                continue
            try:
                new_feature = frame[feature].div( frame[divisor] )
            except ZeroDivisionError:
                print "division is zero. Value skipped."
                continue
            values = new_feature.count()
            corr_poi = frame.corrwith(new_feature)[corr]
            if not np.isnan(corr_poi):
                correlations.append( [feature, divisor, corr_poi, values] )
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    corr_frame = pd.DataFrame(correlations, columns=['feature', 'divisor', 'corr', 'count_instances'])
    return corr_frame


def predict_from_score(y_target, y_score, min_recall, min_precision, verbose=0, return_scores=False):
    '''
    Returns predictions (binary classification) for given prediction scores. The predictions scores get
    modified to meet the requirements for minimum recall and minimum precision.
    
    Atrributes:
    y_target:                  Target classifications
    y_scores:                  Prediction scores. Ether probabilites or decision function output
    min_recall, min_precision: Minimum metric values to be realized.
    verbose:                   Verbosity level (0 - no output, 1 - output)
    return_scores:             Bolean. If True returns the transformed scores instaed of the predictions
    
    Returns new predictions (binary classification 0/1)
    '''
        
    precision, recall, threshold = precision_recall_curve(y_target, y_score)
    size = len(recall)
    for idx in range(size):
        rc = recall[idx]
        if rc >= min_recall:
            pc = precision[idx]
            if pc >= min_precision:
                if idx < size:
                    if recall[idx+1] == rc and precision[idx+1] >= pc:
                        continue
                    else:
                        th = threshold[idx]
                        index = idx
                        break    
            else:
                continue
        else:
            print "Recall =", min_recall, "in combination with precision =", min_precision, "not feasible!"
            return None
    
    if verbose == 1:
        print "Realized metric values:"
        print "Recall:", rc
        print "Precision:", pc
        print "Probability/ decision function:", th
        print "Index:", index
        
    y_score_trans = y_score - th
    y_pred = np.array( map(lambda x: int(x >= 0), y_score_trans) )    
    
    if return_scores == True:
        return y_score_trans
    else:
        return y_pred

def plt_precision_recall_vs_threshold(precision, recall, threshold, title=False):
    '''
    Plots precision and recall versus threshold.
    Expects:
    precisions, recall, threshold values from sklearn.metrics.precision_recall_curve
    Returns plot
    '''
    
    plt.plot(threshold, precision[:-1], "b-", label="Precision", linewidth=1)
    plt.plot(threshold, recall[:-1], "g-", label="Recall", linewidth=1)
    plt.legend(loc="upper center", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.grid(True)
    if title != False:
        plt.title(title)
        
    return None

def plt_precision_vs_recall(precision, recall, title=False):
    plt.plot(recall, precision, "b-", linewidth=1)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    if title != False:
        plt.title(title)
    return None

def plot_roc_curve(fpr, tpr, title=False):
    plt.plot(fpr, tpr, linewidth=1)
    plt.plot([0, 1], [0, 1], 'g--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate', fontsize=12) 
    plt.ylabel('Recall', fontsize=12)    
    plt.grid(True)  
    if title != False:
        plt.title(title)
    return None

def check_Classifiers(X, y, classifiers, imputers=[], scalers=[], names=[], verbose=0, cv=5):
    '''
    '''
    check_df = np.array([])
    if len(names) != len(clf):
        names = [str(x) for x in range(len(clf))]
    for clf, name in zip(classifiers, names):
        for imp in imputers:
            for scl in scalers:
                pipe = Pipeline([ ('impute', imp), ('scale', scl) ])
                X_trans = pipe.fit_transform(X)
                scores = cross_val_score(clf, X_trans, cv=cv)

