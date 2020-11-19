# ENRON fraud detection
Identify persons of interest (POIs) from ENRON financial and email data (educational purpose) 



## Outliers removal



The instances *THE TRAVEL AGENCY IN THE PARK* and *TOTAL* are deleted. 



| THE TRAVEL AGENCY IN THE PARK |  TOTAL |              |
| ----------------------------: | -----: | ------------ |
|                           poi |      0 | 0            |
|                        salary |    NaN | 2.67042e+07  |
|                         bonus |    NaN | 9.73436e+07  |
|           long_term_incentive |    NaN | 4.85219e+07  |
|               deferred_income |    NaN | -2.79929e+07 |
|             deferral_payments |    NaN | 3.20834e+07  |
|                 loan_advances |    NaN | 8.3925e+07   |
|                         other | 362096 | 4.26676e+07  |
|                      expenses |    NaN | 5.2352e+06   |
|                 director_fees |    NaN | 1.39852e+06  |
|                total_payments | 362096 | 3.09887e+08  |
|       exercised_stock_options |    NaN | 3.11764e+08  |
|              restricted_stock |    NaN | 1.30322e+08  |
|     restricted_stock_deferred |    NaN | -7.57679e+06 |
|             total_stock_value |    NaN | 4.3451e+08   |
|                 email_address |    NaN | NaN          |
|                   to_messages |    NaN | NaN          |
|                 from_messages |    NaN | NaN          |
|       from_this_person_to_poi |    NaN | NaN          |
|       from_poi_to_this_person |    NaN | NaN          |
|       shared_receipt_with_poi |    NaN | NaN          |



The instances *BHATNAGAR SANJAY* and *BELFER ROBERT* contain misaligned financial data. After correction the cells look like:

|                   Feature | BELFER ROBERT | BHATNAGAR SANJAY |
| ------------------------: | ------------: | ---------------- |
|                    salary |           NaN | NaN              |
|                     bonus |           NaN | NaN              |
|       long_term_incentive |           NaN | NaN              |
|           deferred_income |           NaN | NaN              |
|         deferral_payments |     -102500.0 | NaN              |
|             loan_advances |           NaN | NaN              |
|                     other |           NaN | 137864.0         |
|                  expenses |           NaN | NaN              |
|             director_fees |        3285.0 | 137864.0         |
|            total_payments |      102500.0 | 15456290.0       |
|   exercised_stock_options |        3285.0 | 2604490.0        |
|          restricted_stock |           NaN | -2604490.0       |
| restricted_stock_deferred |       44093.0 | 15456290.0       |
|         total_stock_value |      -44093.0 | NaN              |



The instance *KAMINSKI WINCENTY J* contains unreasonable value for *from_messages*.  

|           Mail features | KAMINSKI WINCENTY J (before) | KAMINSKI WINCENTY J (after) |
| ----------------------: | ---------------------------: | --------------------------- |
|             to_messages |                       4607.0 | 1211.0                      |
|           from_messages |                      14368.0 | 41.0                        |
| from_this_person_to_poi |                        171.0 | 8.0                         |
| from_poi_to_this_person |                         41.0 | 35.0                        |
| shared_receipt_with_poi |                        583.0 | 740.5                       |



