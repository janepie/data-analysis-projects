Identify Fraud from Enron Email
===


Note: I decided to use Python 3 for this project. The starter code was written for Python 2.7, but as this version is deprecated since the beginning of this year, I adapted the code to Python 3.

## 1. Goal of the project
With this project, I want to analyze the Enron dataset to find Persons of Interest that may be part of the Enron fraud.
The dataset consists of two different kinds of data: financial information about the people working at Enron, and email information that summarizes the email traffic in the company. This is a lot of information I cannot handle manually. To find POIs, I will use machine learning to search for patterns in the dataset. 

The data contains 146 observations with 21 features (one of the features is POI/non POI). 18 of the observations are flagged as POI, 128 are not.

In the dataset still was a 'TOTAL' data point which is useless for my investigation, so I dropped it. One of the observations ('THE TRAVEL AGENCY IN THE PARK') was not a person, and another one with no data included (LOCKHART EUGENE E), I also dropped these.
I had a look on some of the features, and there are still some outliers, but these were Enron's biggest bosses that of course get high salaries and stock options etc. As these are definitely persons of interest, I left them in the data.


## 2. Feature Selection

I decided to not select the features by myself, but to use a selection function for this. With SelectKBest, I can decide how many features I want to use and automatically get my data reduced to the ones that include the most relevant information.
For this, I included almost all features available in the dataset in my data. I had to remove the feature 'email_address' as this address doesn't hold valuable information, it is just a string different for each person. I also checked the features for missing data, and there was some data missing for all of the features:
|feature | missing values | included values |
| --- | --- | --- |
| poi | 0 | 143 |
| salary | 49 | 94 |
| deferral_payments | 105 | 38 |
| total_payments | 20 | 123 |
| loan_advances | 140 | 3 |
| bonus | 62 | 81 |
| restricted_stock_deferred | 126 | 17 |
| deferred_income | 95 | 48 |
| total_stock_value | 18 | 125 |
| expenses | 49 | 94 |
| exercised_stock_options | 42 | 101 |
| other | 52 | 91 |
| long_term_incentive | 78 | 65 |
| restricted_stock | 34 | 109 |
| director_fees | 127 | 16 |
| to_messages | 57 | 86 |
| from_poi_to_this_person | 57 | 86 |
| from_messages | 57 | 86 |
| from_this_person_to_poi | 57 | 86 |
| shared_receipt_with_poi | 57 | 86 |

The feature 'loan_advances' was therefore also excluded as it had only three valid values. There are some more features with lots of missing values, but I leave them in and see if they will be picked by my feature selection algorithm or not.
The three features  'from_poi_to_this_person', 'from_this_person_to_poi' and 'shared_receipt_with_poi' had to be converted to new features as these hold absolute numbers of emails. These absolute numbers should be set in relation to the total number of emails sent or received by the person to get relative information. Therefore, I created the new features 'fraction_from_poi', 'fraction_to_poi' and 'fraction_shared_receipts' and used these instead of the absolute ones.

To have a good range of features, but do not overfit my classifier, I decided to include at least three, but not more than seven in my data. I tested different k's with a Naive Bayes algorithm to find the best one:

| k | Accuracy| Precision | Recall | F1
| -------- | -------- |  -------- |  -------- |  -------- | 
| 3     | 0.84    | 0.49| 0.35 | 0.41 |
| 4     | 0.85    | 0.50| 0.32 | 0.39 |
| 5     | 0.86    | 0.50| 0.33 | 0.39 |
| 6     | 0.86    | 0.52| 0.39 | 0.44 |
| 7     | 0.85    | 0.49| 0.38 | 0.40 |

It is visible that with six features included, the algorithm works best. More or less features have worse test metrics, so I go with k=6.


The algorithm identified these six features as the most useful:
- 'salary'
- 'bonus'
- 'deferred_income'
- 'total_stock_value'
- 'exercised_stock_options'
- 'fraction_to_poi'

All these features have less than 100 missing values, so I think it is okay to use them.

Feature Scaling is not required as I do not use a regression.



## 3. Algorithm Selection

The purpose of the parameter is to decide whether a person is a POI or not. For this, I need a classifier, not an algorithm, as I need a discrete output.
The data already includes labels, so I am able to use supervised learning and don't need to use unsupervised learning.

I tried different supervised learning classifiers to find the best one and tested each (with default parameters) to compare them:

| Algorithm | Accuracy| Precision | Recall | F1
| -------- | -------- |  -------- |  -------- |  -------- | 
| Naive Bayes     | 0.86    | 0.52| 0.39 | 0.44 |
| Decision Tree Classifier     | 0.79 | 0.28 | 0.28 | 0.28 |
| Random Forest     | 0.86 | 0.49 | 0.19 | 0.27 |
| AdaBoost    | 0.82     | 0.35 | 0.28 | 0.31
| K Nearest Neighbors     | 0.88 | 0.69 | 0.26 | 0.37 | 

I also tested a Support Vector Classifier, but that algorithm produced not a single True Positive with 'rbf' kernel and took really long to compute with a linear kernel, so I kicked it out.

The two best-looking algorithms are Naive Bayes and K Nearest Neighbors. KNN is more precise whereas NB is better in Recall (and F1), so with NB I have the best chance to notice a POI in my data and KNN is better in not declaring an innocent person as POI.
As there are no parameters to tune for Naive Bayes, I will try to tune the K Nearest Neighbors algorithm to its best and decide afterwards which classifier I prefer.


## 4. Tuning the algorithm

When tuning the algorithm, I change its parameters to get the best possible output. Here I have to do a trade-off between bias and variance: I want to fit my classifier as good as possible, but I have to be careful to not overfit it.
To have a feeling for the performance of my algorithm, I do the same as when choosing the algorithm: I test it against a testing dataset.

For the K Nearest Neighbors classifier, the two important parameters for tuning are the number of neighbors considered for classifying (n_neighbors default is 5) and the weight function used (weights, 'uniform' (default) or 'distance'). I could also vary the algorithm used to compute the nearest neighbor, but there is an 'auto' option I will use so the best algorithm is automatically chosen.
As I have two different labels, I will use only odd values for n_neighbors so there will always be a clear outcome.

| n_neighbors | weights | Accuracy| Precision | Recall | F1
| -------- | -------- |  -------- |  -------- |  -------- | -------- |
| 1 | uniform | 0.84  | 0.43 | 0.33 | 0.37 |
| 3 | uniform |  0.86 | 0.53 | 0.22 | 0.31 |
| 5 | uniform | 0.88 | 0.69 | 0.26 | 0.37 |
| 7 | uniform |  0.87 | 0.89 | 0.09 | 0.16 |
| 9 | uniform |  all prdictions negative |  |  |  |
| 1 | distance |  0.84  | 0.43 | 0.33 | 0.37 |
| 3 | distance |  0.85 | 0.43 | 0.22 | 0.29 |
| 5 | distance |  0.87 | 0.59 | 0.26 | 0.36 |
| 7 | distance |  0.86 | 0.49 | 0.11 | 0.18 |
| 9 | distance |  0.86 | 0.63 | 0.09 | 0.15 |

The tests show that n_neighbors should not be more than 5 or the Recall gets really low, which means that if there is a POI in the dataset, the chance to find it is really low. 
Weights on 'uniform' performs a little bit better for this data than 'distance'.
The two best parameter sets are 
- n_neighbors = 1 (weights are irrelevant for only one neighbor), better recall
- n_neighbors = 5, distance = 'uniform' (default parameters), better precision

For my investigation, I want the best possible recall to find all the POIs, I will use neither of these but Naive Bayes as it has a better recall.
If there will be False-Positives in my outcome, they wold get a proper manual investigation before they get charged with fraud.


## 5. Validation

Validation of a machine learning algorithm is used to check how reliable the outcome of the analysis is in giving performance indicators. For this, the available training dataset is split in testing and training data to check if the model is overfitted.
To maximize the available testing data, I did a cross-validation with KFold. This function splits my data into training and testing sets several times so I get multiple different data sets and can compare their performance. 
Here, a classic mistake is a data split in two too different sets, i.e. testing data from one person and training data from another person. To avoid this, I used StratifiedKFold to split my data. This function makes sure that all my data sets have the same amount of POI data.
I calculate my evaluation metrics for all folds and return the averages.


## 6. Evaluation Metrics

I get these metrics back:
- Average: 0.87
- Precision: 0.43
- Recall: 0.42
- F1: 0.42

The most important here are precision and recall. 
The precision is the number of true positive predictions divided by all positive predictions, so it shows the possibility of a positive predictions being a true prediction. If this value is high, I can be pretty sure that a prediction flagged as POI is really a POI.
The recall is the number of true positive predictions divided by all positive datapoints (true positives + false negatives), so it shows the possibility of identifying a positive data point if it is there. If this value is high, I can be pretty sure that a person that really is a POI is flagged as POI in my prediction.

For my algorithm, both metrics are in a medium range around 0.5, so I have an average performing classifier which I nevertheless should not trust completely. All findings have to be investigated further, but I can use my predictions as a staring information. 



### Sources:

Classifier and metrics pages in sklearn documentation
https://www.geeksforgeeks.org/python-accessing-all-elements-at-given-list-of-indexes/
https://www.youtube.com/watch?v=HVXime0nQeI