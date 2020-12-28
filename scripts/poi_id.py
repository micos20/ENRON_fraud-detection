#!/usr/bin/python

import pandas as pd
import numpy as np
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
sys.path.append("../tools/")

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

'''
#############################
### Task 3: Create new feature(s)
# Automatically create new features and correlate with 'poi'
#corr_list = payment_features + stock_features + mail_features
#corr_df = helper.correlate(data_Frame, 'poi', feature_list=corr_list)
#print corr_df[ corr_df['corr'] > 0.15 ]

# Add email features (POI rates for send and received emails)
data_Frame['toPOI_rate']                                = data_Frame['from_this_person_to_poi'].div(data_Frame['from_messages'])
data_Frame['fromPOI_rate']                              = data_Frame['from_poi_to_this_person'].div(data_Frame['to_messages'])

# Create new financial features
data_Frame['bonus_deferral_payments_rate']              = data_Frame['bonus'].div( data_Frame['deferral_payments'] )
data_Frame['rest_stock_deferral_payments_rate']         = data_Frame['restricted_stock'].div( data_Frame['deferral_payments'] )
data_Frame['exer_stock_options_deferral_payments_rate'] = data_Frame['exercised_stock_options'].div( data_Frame['deferral_payments'] )

data_Frame['long_term_incentive_total_payments_rate']   = data_Frame['long_term_incentive'].div( data_Frame['total_payments'] )
data_Frame['bonus_total_payments_rate']                 = data_Frame['bonus'].div( data_Frame['total_payments'] )
data_Frame['exer_stock_options_total_payments_rate']    = data_Frame['exercised_stock_options'].div( data_Frame['total_payments'] )


# Create feature list
payment_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'other', 'expenses', 'director_fees', 'total_payments']
stock_features   = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
mail_features    = ['from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi']
extra_features = ['bonus_deferral_payments_rate', 'rest_stock_deferral_payments_rate', 'exer_stock_options_deferral_payments_rate',
                  'long_term_incentive_total_payments_rate', 'bonus_total_payments_rate', 'exer_stock_options_total_payments_rate',
                  'toPOI_rate', 'fromPOI_rate']

feature_list = ['poi'] + payment_features + stock_features + mail_features + extra_features

# Create data frame containing all features to be used
data_Frame = data_Frame[ feature_list ]


### Store to my_dataset for easy export below.
my_dataset = data_Frame.transpose().to_dict()

print data_Frame.head()


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
'''