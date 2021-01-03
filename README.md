

# ENRON fraud detection

Identify persons of interest (POIs) from ENRON financial and email data (educational purpose) 



## 1. Project goal

This project is about using machine learning (ML) to identify Persons of Interest (POI) involved in the Enron scandal which surfaced end of 2001. By systematic usage of accounting loopholes and poor financial reporting, involved executives and employees were able to hide billions of dollars from failed deals and projects. As a result Enron went bankrupt and many people and share holder lost their money and pensions. Machine learning (binary classification) is used to identify POIs by financial and email data. Supervised ML is particularly useful in this area as it can identify patterns in data automatically and relate them to a given class (poi/ non-poi). In contrast, conservative coding would require extreme complex rules and algorithms for this classification task.    

#### The dataset

The present dataset is taken from the [*Enron Email Dataset*]( https://www.cs.cmu.edu/~./enron/) and comprises financial and email data. There are 146 data points (employees and executives from Enron). 18 people are identified as POI and 128 as non-POI. There are 20 features in the dataset. 10 of them are payment features, 4 are stock features (14 financial features) and there are 6 email features. From the output below you can see that all features show missing data. *loan_advances* does only have only 4 *non-null* values and *director_fees*, *restricted_stock_deferred* only 17, respectively 18 *non-null* values. 

```python
data_Frame.info()
<class 'pandas.core.frame.DataFrame'>
Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
Data columns (total 21 columns):
poi                          146 non-null int64
salary                       95 non-null float64
bonus                        82 non-null float64
long_term_incentive          66 non-null float64
deferred_income              49 non-null float64
deferral_payments            39 non-null float64
loan_advances                4 non-null float64
other                        93 non-null float64
expenses                     95 non-null float64
director_fees                17 non-null float64
total_payments               125 non-null float64
exercised_stock_options      102 non-null float64
restricted_stock             110 non-null float64
restricted_stock_deferred    18 non-null float64
total_stock_value            126 non-null float64
email_address                111 non-null object
to_messages                  86 non-null float64
from_messages                86 non-null float64
from_this_person_to_poi      86 non-null float64
from_poi_to_this_person      86 non-null float64
shared_receipt_with_poi      86 non-null float64
dtypes: float64(19), int64(1), object(1)
memory usage: 25.1+ KB
```

#### Outliers detection

In order to take a closer look to the dataset I created a [histogram plot](./images/features_histogram_plots.png) for all features. Almost all features show extreme outliers. Checking the *bonus* feature for values above 10M USD reveals a data point called *TOTAL*, which is the sum of each column (feature). I delete *TOTAL* from the dataset. 

By cross checking the list of names in the dataset I found a data point named *THE TRAVEL AGENCY IN THE PARK*. I remove it as well as the data is not related to a specific person and it only contains two features (*other*, *total_payments*). The feature *total_payments* is the sum of all payment features (*salary*, *bonus*, *long_term_incentive*, *deferred_income*, *deferral_payments*, *other*, *expenses*, *director_fees*) and *total_stock_value* is the sum of stock features (*exercised_stock_options*, *restricted_stock*, *restricted_stock_deferred*). Comparing the sum of these features to the total values reveals a mismatch for the two instances *BHATNAGAR SANJAY* and *BELFER ROBERT*. The values weren't imported correctly from the [original financial feature list](./data/enron61702insiderpay.pdf). The corrected data can be seen in the table below. 

|          Feature          | BELFER ROBERT | BHATNAGAR SANJAY |
| :-----------------------: | ------------: | ---------------: |
|          salary           |           NaN |              NaN |
|           bonus           |           NaN |              NaN |
|    long_term_incentive    |           NaN |              NaN |
|      deferred_income      |     -102500.0 |              NaN |
|     deferral_payments     |           NaN |              NaN |
|       loan_advances       |           NaN |              NaN |
|           other           |           NaN |              NaN |
|         expenses          |        3285.0 |         137864.0 |
|       director_fees       |      102500.0 |              NaN |
|      total_payments       |        3285.0 |         137864.0 |
|  exercised_stock_options  |           NaN |       15456290.0 |
|     restricted_stock      |       44093.0 |        2604490.0 |
| restricted_stock_deferred |      -44093.0 |       -2604490.0 |
|     total_stock_value     |           NaN |       15456290.0 |

At last I check the email features. The instance *KAMINSKI WINCENTY J* shows unreasonable value for the feature *from_messages*. I actually do not believe this guy send more than 14k emails and only received 4.6k.   I use median mail values for *Kaminski* as the current values are not trustworthy to me. The result can be seen in the table below.   

|           Mail features | KAMINSKI WINCENTY J (before) | KAMINSKI WINCENTY J (after) |
| ----------------------: | ---------------------------: | --------------------------- |
|             to_messages |                       4607.0 | 1211.0                      |
|           from_messages |                      14368.0 | 41.0                        |
| from_this_person_to_poi |                        171.0 | 8.0                         |
| from_poi_to_this_person |                         41.0 | 35.0                        |
| shared_receipt_with_poi |                        583.0 | 740.5                       |

The image below shows the histogram plots of all corrected features. In addition I created [box plots](./images/features_box_plots_corrected.png) for all features to assess basic statistics (mean values, standard dev., ...) quickly. 

![Corrected features](./images/features_histogram_plots_corrected.png)

It is obvious that there are still many outliers in the dataset. As extreme values might be an indication for fraud or any other irregular operation I'll keep the remaining data as is.



## 2. Feature selection

There are 20 features provided with the current dataset. The feature *email_address* does not seem to be a good indicator whether a person is a POI or not. I drop this feature. As we've seen above *loan_advances* is available for four instances only. This seems to me too few information, so I drop it as well. 

#### New features

Before starting with the selection process to find the best features for a chosen algorithm I like to incorporate a few new features. The features *to_messages* and *from_messages* are the number of mails a person received (*to*) and sent (*from*) from/ to others. The sheer number of e-mails cannot be an indicator of fraud. But maybe the ratio of mails sent to a poi divided by the number of sent messages. Therefore I created two new features, the *toPOI_rate* and *fromPOI_rate*.

In order to create further new features I divide each feature by all other features and calculate the pearson correlation coefficient with the *poi* column. The best correlations can be seen in the table below. 

```
                    feature              divisor      corr  count
0                     bonus    deferral_payments  0.665900     21
1   from_this_person_to_poi    deferral_payments  0.650054     22
2   shared_receipt_with_poi    deferral_payments  0.551253     22
3          restricted_stock    deferral_payments  0.516681     26
4   exercised_stock_options    deferral_payments  0.493101     32
5   from_poi_to_this_person    deferral_payments  0.471403     22
6         total_stock_value    deferral_payments  0.447212     36
7            total_payments    deferral_payments  0.441996     37
9   exercised_stock_options        from_messages  0.373662     67
11                 expenses    deferral_payments  0.356435     22
13  from_this_person_to_poi        from_messages  0.336790     86
14              to_messages    deferral_payments  0.336584     22
15      long_term_incentive       total_payments  0.326070     65
17                   salary    deferral_payments  0.304583     26
18        total_stock_value        from_messages  0.301027     81
19                    bonus       total_payments  0.297819     81
20                    bonus        from_messages  0.297435     61
23  exercised_stock_options       total_payments  0.274168     85
24          deferred_income  long_term_incentive  0.271228     23
25  shared_receipt_with_poi          to_messages  0.260937     86
```

Based on this data I created new features as can be seen in the table below. This is actually just an experiment. I want to see how these auto-features work.  

| New feature                                 | Numerator                 | Denominator         |
| ------------------------------------------- | ------------------------- | ------------------- |
| *bonus_deferral_payments_rate*              | *bonus*                   | *deferral_payments* |
| *rest_stock_deferral_payments_rate*         | *restricted_stock*        | *deferral_payments* |
| *exer_stock_options_deferral_payments_rate* | *exercised_stock_options* | *deferral_payments* |
| *long_term_incentive_total_payments_rate*   | *long_term_incentive*     | *total_payments*    |
| *bonus_total_payments_rate*                 | *bonus*                   | *total_payments*    |
| *exer_stock_options_total_payments_rate*    | *exercised_stock_options* | *total_payments*    |

#### Imputation and feature scaling

As we've seen above there is a lot of missing data in our dataset. During the course of this project I checked different imputation strategies. For the finally selected POI identifier I decided to fill the financial features with 0.0. For the email features I use the median for all missing values. The same strategy is applied to the newly created features. For *toPOI_rate* and *fromPOI_rate* I use the median and for the extra financial features I use 0.0. This is imputation strategy 4 (*impute_04*) from the *poi_id.py* script.    

The final AI model is a Support Vector Classifier. Support Vector Machines are sensitive to feature scales, so the final features are scaled using a Power Transformer applying the *yeo-johnson* method. I also used the Standard Scaler and the Robust Scaler but the algorithm works best with the Power Transformer. This might be caused by the many outliers still in our dataset.

#### Selection of features

For feature selection I use cross validated recursive feature elimination (RFECV) method. An estimator is required to run RFECV method, so we will come up with different sets of features for each investigated classifier. RFECV provides feature rankings. All selected features will have rank 1. The remaining features are numbered in accordance to their importance. The results of feature ranking for a Support Vector classifier (SVC) and a Stochastic Gradient Descent classifier (SGD) are shown in the table below.  

***Ranking of features by RFECV method***

| Feature                                   | SVC rank | SGD rank |
| ----------------------------------------- | :------: | -------- |
| expenses                                  |    1     | 1        |
| total_stock_value                         |    1     | 1        |
| exercised_stock_options                   |    1     | 1        |
| toPOI_rate                                |    1     | 1        |
| bonus                                     |    1     | 7        |
| exer_stock_options_deferral_payments_rate |    1     | 1        |
| rest_stock_deferral_payments_rate         |    1     | 3        |
| bonus_deferral_payments_rate              |    1     | 1        |
| deferred_income                           |    1     | 1        |
| shared_receipt_with_poi                   |    1     | 1        |
| salary                                    |    2     | 1        |
| exer_stock_options_total_payments_rate    |    3     | 8        |
| other                                     |    4     | 1        |
| deferral_payments                         |    5     | 10       |
| long_term_incentive_total_payments_rate   |    6     | 11       |
| restricted_stock_deferred                 |    7     | 9        |
| long_term_incentive                       |    8     | 1        |
| bonus_total_payments_rate                 |    9     | 1        |
| from_this_person_to_poi                   |    10    | 5        |
| total_payments                            |    11    | 13       |
| restricted_stock                          |    12    | 6        |
| from_poi_to_this_person                   |    13    | 12       |
| fromPOI_rate                              |    14    | 2        |
| director_fees                             |    15    | 4        |

For the SVM classifier the selected features are *expenses, total_stock_value, exercised_stock_options, toPOI_rate, bonus, exer_stock_options_deferral_payments_rate, rest_stock_deferral_payments_rate, bonus_deferral_payments_rate, deferred_income* and *shared_receipt_with_poi*. It is interesting that 4 out of the 10 features are newly created features.

For the SGD model we have a different set of features as can be seen above. But again, 4 out of 12 features are newly created ones. 



## 3. Model selection and tuning

I choose two different algorithms, a Support Vector Classifier (SVC) and a Stochastic Gradient Descent Classifier (SGD). In order to tune the algorithms I use a cross validated randomized search algorithm (RandomizedSearchCV) with 5000 iterations to find the best hyper parameters. 

Tuning the model's parameters helps optimizing the algorithm to perform better. We use the training data to optimize these parameters. Optimization to the training data may lead to over-fitting, so that the resulting model might not generalize well on the test set or in real life operation. In order to minimize over-fitting I use a cross validation approach to tune the parameters of the model.     

#### Support Vector Machine    

The following parameters are tuned:

```python
`param_distributions = {`
            `'kernel': ['linear', 'rbf'],`
            `'C': stats.uniform(0.1, 10000),`
            `'gamma': stats.expon(scale=1.0),`
        }`
```

I try a linear and *RBF* kernel. The regulation parameter C is randomly chosen between 0.1 and 10000. The last parameter is gamma for witch I use a exponential distribution function. The best parameters for the SVC model are: 

| Parameters of SVM classifier | Value       |
| ---------------------------- | ----------- |
| kernel                       | rbf         |
| C                            | 1143.11...  |
| gamma                        | 0.001136... |

#### Stochastic Gradient Descent

For the SGD classifier I perform a parameter search for each of the four different options of the learning_rate parameters, *optimal, constant, invscaling* and *adaptive*. I only document the tuned parameters of the best learning_rate option, in this case *optimal*. The following hyper-parameters are tuned:

```python
param_distributions = {
    'penalty': ['l2', 'l1'],
    'alpha': stats.uniform(10**(-6), 50),
}
```
The results of the Randomized Search is shown in the table below:

| Parameters of SGD classifier | Value      |
| ---------------------------- | ---------- |
| penalty                      | *l2*       |
| alpha                        | 6.75588... |
| learning_rate                | *optimal*  |

#### Comparison of models

In order to compare the performance of the SGD and SVC model I plot the precision vs recall curve for each classifier. In addition I add the untuned curves as well to see how parameter tuning changed the performance of the models. The precision vs recall curves for the tuned and untuned models can be seen in the image below.

![](./images/precision_vs_recall_on_train_set.png)

For the SGD model we can observe a great improvement for the tuned model (green vs yellow). For the SVM classifier (red vs blue) the improvement is not so significant. Overall, the SVC classifier seems to work slightly better on the training set as the tuned SGD model. In the next section, we'll see how the models perform on the test set.

## 4. Validate and evaluate





## 5. Conclusion

Importance of size of test set.



   



 