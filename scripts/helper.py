#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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