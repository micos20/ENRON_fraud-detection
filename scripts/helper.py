#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

def repl_NaN(val):
    '''
    Replaces string 'NaN' by numpy.NaN type and boolean type by integer 1 for 'True' and 0 for 'False'
    '''
    if val == 'NaN':
        val = np.NaN
    if isinstance(val, bool):
        val = int(val)
    return val


def crt_plot(data, features, type='box', figsize=(20,15), shape=(1,1), log=[], bins=10, sort=True, color='orange', save=False, show=True):
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
    if show:
        plt.show()
    
    return None

# 
def correlate(frame, corr, feature_list=None):
    '''
    Correlates feature combinations of data frame columns with selected feature. 
    It automatically creates new features by dividing every feature in 'frame' by all other features in 'feature_list'.
    New features are correlated to 'corr' column and the results are returned as pandas data frame in the form:
    'feature', 'divisor', 'pearson correlation', 'number of data points'
    '''
    correlations = []   # store correlation results
    if feature_list == None:
        feature_list = frame.columns
    for feature in feature_list:
        for divisor in feature_list:
            if feature == divisor:
                continue   # skip correlations with same column
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
    corr_frame = pd.DataFrame(correlations, columns=['numerator', 'denominator', 'corr', 'count_instances'])
    
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

def plt_precision_vs_recall(precision, recall, title=False, save=False, **kwargs):
    plt.plot(recall, precision, "b-", linewidth=5., **kwargs)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    if 'label' in kwargs.keys():
        plt.legend(fontsize=20)
    if title != False:
        plt.title(title, fontsize=30)
    if save != False:
        plt.savefig(save, dpi=75, bbox_inches='tight', pad_inches=0.5)
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

class flex_classifier(BaseEstimator):
    '''
    '''
    def __init__(self, classifier, min_precision=0.0, min_recall=0.0, maximize='recall', threshold=0.0):
        self.classifier = clone(classifier)
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.maximize = maximize
        self.threshold = threshold
        self.feasibiliy = None
        #self.classifier.kernel = kernel
        #self.classifier.gamma = gamma
        #self.classifier.C = C
        if type(classifier) == RandomForestClassifier:
            self.method = 'predict_proba'
        else:
            self.method = 'decision_function'
        
    def fit(self, X, y=None):
        self.classifier.fit(X, y)
        #self.threshold = self.det_threshold(X, y)
        return self
    
    def predict(self, X):
        if self.method == 'decision_function':
            df_h = self.classifier.decision_function(X) - self.threshold
        else:
            df_h = self.classifier.predict_proba(X)[:,1] - (0.499999 - self.threshold)
        
        y_h = df_h >= self.threshold
        return y_h.astype('uint8')

    def det_threshold(self, X, y, replace=True):
        '''
        Determine best theshold to achieve given precision and recall values
        '''
        if self.method == 'predict_proba':
            scores = cross_val_predict(self.classifier, X, y, cv=6, method=self.method)[:,1] - 0.499999
        else:
            scores = cross_val_predict(self.classifier, X, y, cv=6, method=self.method)
        
        ranked = np.sort(scores)
        predict = (scores >= 0.).astype('uint8')
        recall = recall_score(y, predict)
        precision = precision_score(y, predict)
 
        if recall < self.min_recall and precision < self.min_precision:
            self.feasibiliy = False
            return 0.0, recall, precision
        elif recall == 1.0 and precision >= self.min_precision:
            self.feasibiliy = True
            return 0.0, recall, precision
        else:
            self.feasibiliy = True
        
        threshold_matrix = [ [self.threshold, recall, precision], ]
        if self.maximize == 'recall':
            for threshold in (score for score in ranked[::-1] if score < 0 ):
                predict = (scores-threshold >= 0.).astype('uint8')
                recall = recall_score(y, predict)
                precision = precision_score(y, predict)
                threshold_matrix.append([threshold, recall, precision])
            #print(np.array(threshold_matrix))
            threshold, recall, precision = threshold_matrix[0]
            for threshold_, recall_, precision_ in threshold_matrix[1:]:
                if precision_ < self.min_precision:
                    break
                if recall_ > recall :
                    threshold, recall, precision = threshold_, recall_, precision_ 
        elif self.maximize == 'precision':
            threshold = self.threshold
            # This feature is not yet implemented
        
        if replace:
            self.threshold = threshold
        
        return threshold, recall, precision

def get_features(features, support):
    '''
    Extract features from from 'feature' list by boolean 'support' list.

    '''
    f_list = []
    for feature, choise in zip(features, support):
        if choise: 
            f_list.append(feature)
    return f_list