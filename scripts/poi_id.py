#!/usr/bin/python

import pandas as pd
import numpy as np
import scipy.stats as stats
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
sys.path.append("../tools/")
from sklearn.metrics import confusion_matrix, precision_recall_curve, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

IMAGES = '../images/'
DATA   = '../data/'

#############################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
payment_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments']
stock_features   = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
mail_features    = ['from_messages', 'to_messages', 'from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi']

features = payment_features + stock_features + mail_features

### Load the dictionary containing the dataset
with open(DATA+"final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# Column names
columns = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
           'email_address', 'to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi'] 
data_Frame = pd.DataFrame(data_dict).transpose()
data_Frame = data_Frame[ columns ]    

# Replace 'NaN' string with numpy.NaN and change POI from boolean to integer (poi: 1, no_poi: 0)
data_Frame = data_Frame.applymap(helper.repl_NaN)

print "Understanding the dataset"
print "-------------------------"

# Determine number of data points
n = len(data_dict.keys())
print "Number of data points:", n
# Number of pois/ non-pois
nbr_pois = data_Frame.poi.sum()
print "Number of pois/ non-pois:", nbr_pois, "/", n-nbr_pois
# Number of features
print "Number of features:", len(columns) - 1
# Further info regarding features
print "Further info regarding features:"
print data_Frame.info()

#############################
### Task 2: Remove outliers
# Create histograms for all features and save the image
data_Frame.hist(bins=20,figsize=(20,15))
plt.tight_layout()
plt.savefig(IMAGES+"features_histogram_plots", dpi='figure')
# plt.show()
print "Histogram plot created and saved. See './images/features_histogram_plots.png'."
print
print "Bonus above 10M USD"
print data_Frame[ data_Frame['bonus'] >= 10000000 ]

# Dropping the 'TOTAL' instance
data_Frame.drop(labels='TOTAL', inplace=True)
# Drop 'THE TRAVEL AGENCY IN THE PARK' instance. This is an agency and not a real person. In addition there is not much data available for this instance.
data_Frame.drop(labels='THE TRAVEL AGENCY IN THE PARK', inplace=True)
print "Deleted the instances 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'."

# Checking the total stock and payment feature
data_Frame_eval = data_Frame.fillna(value=0.0)
data_Frame_eval['eval_payments'] = data_Frame_eval['salary'] + data_Frame_eval['bonus'] + data_Frame_eval['long_term_incentive'] + data_Frame_eval['deferred_income'] + data_Frame_eval['deferral_payments'] + data_Frame_eval['loan_advances'] + data_Frame_eval['other'] + data_Frame_eval['expenses'] + data_Frame_eval['director_fees'] - data_Frame_eval['total_payments']

print
print "Check 'total_payments' feature:"
print data_Frame_eval[ data_Frame_eval['eval_payments'] != 0.0 ][['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'eval_payments', 
           'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']]

data_Frame_eval['eval_stock'] = data_Frame_eval['exercised_stock_options'] + data_Frame_eval['restricted_stock'] + data_Frame_eval['restricted_stock_deferred'] - data_Frame_eval['total_stock_value']

print
print "Check 'total_stock_value' feature:"
print data_Frame_eval[ data_Frame_eval['eval_stock'] != 0.0 ][['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'eval_payments', 
           'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'eval_stock'] ]

# Correcting Robert Belfer instance
print "Correcting 'Robert Belfer' instance."
data_Frame.loc['BELFER ROBERT', 'restricted_stock'] = 44093.0
data_Frame.loc['BELFER ROBERT', 'restricted_stock_deferred'] = -44093.0
data_Frame.loc['BELFER ROBERT', 'exercised_stock_options'] = np.NaN
data_Frame.loc['BELFER ROBERT', 'total_stock_value'] = np.NaN
data_Frame.loc['BELFER ROBERT', 'director_fees'] = 102500.0 
data_Frame.loc['BELFER ROBERT', 'deferred_income'] = -102500.0
data_Frame.loc['BELFER ROBERT', 'deferral_payments'] = np.NaN
data_Frame.loc['BELFER ROBERT', 'expenses'] = 3285.0
data_Frame.loc['BELFER ROBERT', 'total_payments'] = 3285.0

# Correcting BHATNAGAR SANJAY instance
print "Correcting 'BHATNAGAR SANJAY' instance."
data_Frame.loc['BHATNAGAR SANJAY', 'other'] = np.NaN
data_Frame.loc['BHATNAGAR SANJAY', 'expenses'] = 137864.0
data_Frame.loc['BHATNAGAR SANJAY', 'director_fees'] = np.NaN
data_Frame.loc['BHATNAGAR SANJAY', 'total_payments'] = 137864.0
data_Frame.loc['BHATNAGAR SANJAY', 'exercised_stock_options'] = 15456290.0
data_Frame.loc['BHATNAGAR SANJAY', 'restricted_stock'] = 2604490.0
data_Frame.loc['BHATNAGAR SANJAY', 'restricted_stock_deferred'] = -2604490.0
data_Frame.loc['BHATNAGAR SANJAY', 'total_stock_value'] = 15456290.0

print
print "Corrected instances 'BELFER ROBERT' and 'BHATNAGAR SANJAY':"
print data_Frame.loc[ ['BELFER ROBERT', 'BHATNAGAR SANJAY'] ][['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
           'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']].transpose()

# Checked instance 'Kaminski'. It doesn't look correct. He clearly didn't send 14k mails and just received 4600.
print
print "Mail features: 'from_messages' >= 5000"
print data_Frame[ data_Frame['from_messages'] >= 5000 ][['poi']+mail_features].transpose()

# I'll use median mail values for Kaminski as the current values are not trustworthy to me. 
# Unfortunately I cannot resolve where or how the mail data was generated.
data_Frame.loc['KAMINSKI WINCENTY J', mail_features] = data_Frame[mail_features].median()
print
print "Median values for mail_features for the instance ''KAMINSKI WINCENTY J'"
print data_Frame.loc['KAMINSKI WINCENTY J', mail_features]

# Create histograms for corrected features and save the image
data_Frame.hist(bins=20,figsize=(20,15))
plt.tight_layout()
plt.savefig(IMAGES+"features_histogram_plots_corrected", dpi='figure')
print
print "Histogram plots created and saved. See './images/features_histogram_plots_corrected.png'."

# Create box plots of corrected features
# Logarytmic scale for some box plots
log_list = ['from_messages', 'from_this_person_to_poi', 'total_payments', 'total_stock_value', 'other', 'restricted_stock', 'exercised_stock_options']
helper.crt_plot(data_Frame, ["poi"] + features, shape=(5,4), log=log_list, sort=False, save=IMAGES+"features_box_plots_corrected", show=False)
print
print "Box plots created and saved. See './images/features_box_plots_corrected.png'."


#############################
### Task 3: Create new feature(s)

# Remove 'loan_advances' from features list
features.remove('loan_advances')
payment_features.remove('loan_advances')

# Add email features (POI rates for send and received emails)
data_Frame['toPOI_rate']                                = data_Frame['from_this_person_to_poi'].div(data_Frame['from_messages'])
data_Frame['fromPOI_rate']                              = data_Frame['from_poi_to_this_person'].div(data_Frame['to_messages'])

# Automatically create new features and calculate pearson correlation coef.
# corr_list = list(feature_list)
corr_df = helper.correlate(data_Frame, 'poi', feature_list=features)
print
print corr_df[ corr_df['corr'] > 0.25 ]

# Create new financial features
data_Frame['bonus_deferral_payments_rate']              = data_Frame['bonus'].div( data_Frame['deferral_payments'] )
data_Frame['rest_stock_deferral_payments_rate']         = data_Frame['restricted_stock'].div( data_Frame['deferral_payments'] )
data_Frame['exer_stock_options_deferral_payments_rate'] = data_Frame['exercised_stock_options'].div( data_Frame['deferral_payments'] )

data_Frame['long_term_incentive_total_payments_rate']   = data_Frame['long_term_incentive'].div( data_Frame['total_payments'] )
data_Frame['bonus_total_payments_rate']                 = data_Frame['bonus'].div( data_Frame['total_payments'] )
data_Frame['exer_stock_options_total_payments_rate']    = data_Frame['exercised_stock_options'].div( data_Frame['total_payments'] )

extra_finance    = ['bonus_deferral_payments_rate', 'rest_stock_deferral_payments_rate', 'exer_stock_options_deferral_payments_rate',
                    'long_term_incentive_total_payments_rate', 'bonus_total_payments_rate', 'exer_stock_options_total_payments_rate']
extra_mail       = ['toPOI_rate', 'fromPOI_rate']

# Imputation strategies
# ---------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# I'd like to explore different strategies filling NAN values
# Imputer_01:
# Applies median to mail and extra NAN features and 0 to financial NAN features
impute_01 = ColumnTransformer(
     [('finance_data',  SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0), payment_features+stock_features),
      ('mail_data',     SimpleImputer(missing_values=np.nan, strategy='median'), mail_features),
      ('extra_finance', SimpleImputer(missing_values=np.nan, strategy='median'), extra_finance),
      ('extra_mail',    SimpleImputer(missing_values=np.nan, strategy='median'), extra_mail)],
     remainder='passthrough')
# Imputer_02:
# Applies median to all NAN features 
impute_02 = SimpleImputer(strategy='median')
# Imputer_03:
# Applies 0.0 to all NAN features
impute_03 = SimpleImputer(strategy='constant', fill_value=0.0)
# Imputer 04
# Applies 0.0 to financial and extra_finance featurtes and median to 'mail_features' and 'extra_mail'
impute_04 = ColumnTransformer(
     [('finance_data',  SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0), payment_features+stock_features),
      ('mail_data',     SimpleImputer(missing_values=np.nan, strategy='median'), mail_features),
      ('extra_finance', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0), extra_finance),
      ('extra_mail',    SimpleImputer(missing_values=np.nan, strategy='median'), extra_mail)],
     remainder='passthrough')
# Imputer 05
# Applies mean values to all NAN
impute_05 = SimpleImputer(strategy='mean')
imputers = [impute_01, impute_02, impute_03, impute_04, impute_05]

# Feature scaling
# Define different scaling strategies
robust_scl = RobustScaler()
std_scl    = StandardScaler()
power_scl  = PowerTransformer(method='yeo-johnson')
scalers = [robust_scl, std_scl, power_scl]
scaler_names = ['NO SCALING', 'ROBUST SCALER', 'STANDARD SCALER', 'POWER SCALER']

# Build pipelines for imputation and scaling
pipe_44 = Pipeline([ ('impute_04', impute_04), ('scale', power_scl) ])

features.remove('to_messages')
features.remove('from_messages')
mail_features.remove('to_messages')
mail_features.remove('from_messages')
features = features + extra_finance + extra_mail

# Divide data into features and labels 
y = data_Frame['poi'].copy().astype(np.uint8)
X = data_Frame[features].copy()
# Split data into training and test set using stratified splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=77, stratify=y) 

#X_train_41 = pipe_41.fit_transform(X_train);
X_train_44 = pipe_44.fit_transform(X_train);
X_train_44_df = pd.DataFrame(X_train_44, columns=features)

# Feature reduction using SelectKBest
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, SelectFdr, SelectFpr
KBest = SelectKBest(k=24)
KBest.fit_transform(X_train_44, y_train)
KBest_scores_df = pd.DataFrame(KBest.scores_, index=features)
print
print "Feature reduction"
print "-----------------"
print "SelectKBest scores:"
print KBest_scores_df.sort_values(0, ascending=False)
print

from sklearn import svm
SVC_lin = svm.SVC(gamma='auto', kernel='linear')
print "ROC AUC scores for SelectKBest features:"
for n in range(1, len(features)+1):
    KBest = SelectKBest(k=n)
    X_train_44_ = KBest.fit_transform(X_train_44, y_train)
    y_scores = cross_val_predict(SVC_lin, X_train_44_, y_train, cv=6, method='decision_function')
    print roc_auc_score(y_train, y_scores), "ROC AUC for", n, "features."
print

# Feature reduction using RFE
from sklearn.feature_selection import RFE, RFECV
RFE_1 = RFE(SVC_lin, n_features_to_select=1).fit(X_train_44, y_train)
score_table = pd.DataFrame(zip(X_train_44_df.columns.to_list(), RFE_1.ranking_))
print "RFE ranking of features"
print score_table.sort_values(1, ascending=True)
print

print "ROC AUC scores for RFE best features:"
for n in range(1, len(features)+1):
    X_train_44_ = RFE(SVC_lin, n_features_to_select=n).fit_transform(X_train_44, y_train)
    y_scores = cross_val_predict(SVC_lin, X_train_44_, y_train, cv=6, method='decision_function')
    print roc_auc_score(y_train, y_scores), "ROC AUC for", n, "features."
print

# Plot precision vs recall curve for 10 best features leading to highest ROC
plt.clf()
RFE_10 = RFE(SVC_lin, n_features_to_select=10).fit(X_train_44, y_train)
X_train_44_10 = RFE_10.transform(X_train_44)
y_scores = cross_val_predict(SVC_lin, X_train_44_10, y_train, cv=6, method='decision_function')
print roc_auc_score(y_train, y_scores), "ROC AUC for 10 best features."
precision, recall, proba = precision_recall_curve(y_train, y_scores)
helper.plt_precision_vs_recall(precision, recall, title='SVC 10 best features', save=IMAGES+"RFE_precision_vs_recall.png", label='untuned_SVC', color='blue')
print "Image 'RFE_precision_vs_recall.png' saved."
print


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below. 
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Support Vector Classifier
clf = svm.SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Random search to find best set of features
param_distributions = {
            'kernel': ['linear', 'rbf'],
            'C': stats.halfnorm(0.1, 10000),
            'gamma': stats.expon(scale=1.0),
        }
SVC_RandSearch = RandomizedSearchCV(clf, param_distributions, cv=6, n_iter=5000, scoring='roc_auc', iid=False, verbose=1, n_jobs=8, random_state=77)
SVC_RandSearch.fit(X_train_44_10, y_train)

y_scores = cross_val_predict(SVC_RandSearch.best_estimator_, X_train_44_10, y_train, cv=6, method='decision_function')
print roc_auc_score(y_train, y_scores), "ROC AUC curve for tuned SVM classifier"
precision, recall, proba = precision_recall_curve(y_train, y_scores)
helper.plt_precision_vs_recall(precision, recall, title='ROC AUC curve for tuned SVM classifier', save=IMAGES+"ROC_AUC_curve_tuned_SVC.png", label='tuned_SVC', color='red')

'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
'''