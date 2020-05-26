#!/usr/bin/python

import sys
from time import time
import pickle
from operator import itemgetter 
from statistics import mean 
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees', 'to_messages', 'fraction_from_poi', 'from_messages', 
                 'fraction_to_poi', 'fraction_shared_receipts'] # You will need to use more features, not: email addresses

                 
feat_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
        


### Task 2: Remove outliers

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)


for feature in feat_list:
    i = 0
    y = 0
    for list in data_dict.values():
        if list[feature] == 'NaN':
            i += 1
        else:
            y += 1
    print('|', feature, '|', i, '|', y, '|')
### Task 3: Create new feature(s)


# adds a new feature to the data where the absolute feature is divided by 
# the divident feature
# code from lessons (adapted)
def calculate_fraction(data_dict, absolute_feature, divident_feature, new_feature): 
    submit_dict = {}   
    for name in data_dict:

        data_point = data_dict[name]
    
        abs_feat = data_point[absolute_feature]
        div = data_point[divident_feature]
        try:
            fraction = float(abs_feat) / div
        except:
            fraction = 0.
        data_point[new_feature] = fraction
        submit_dict[name] = data_point
    return submit_dict


# now adding the new features
data_dict = calculate_fraction(data_dict, "from_poi_to_this_person", "to_messages", "fraction_from_poi")
data_dict = calculate_fraction(data_dict, "from_this_person_to_poi", "from_messages", "fraction_to_poi")
data_dict = calculate_fraction(data_dict, "shared_receipt_with_poi", "to_messages", "fraction_shared_receipts")

### Store to my_dataset for easy export below.
my_dataset = data_dict




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)



labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=6)
features = selector.fit_transform(features, labels)
# creation of list of used features for use with tester.py
feat_index =  selector.get_support(indices = True)
my_features_list = ['poi']
my_features_list.extend(list(itemgetter(*feat_index + 1)(features_list)))
print('Used features:', my_features_list)


### Task 4: Try a varity of classifiers


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, data_dict, my_features_list, folds = 1000)

# Validation


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
skf = StratifiedKFold(n_splits=5)
accuracy = []
precision = []
recall = []
f1 = []
for train_indices, test_indices in skf.split(features, labels):
    features_train = [features[ii] for ii in train_indices]
    features_test  = [features[ii] for ii in test_indices]
    labels_train   = [labels[ii] for ii in train_indices]
    labels_test    = [labels[ii] for ii in test_indices]
    
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy.append(accuracy_score(labels_test, pred))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))
    f1.append(f1_score(labels_test, pred))
    
print('Accuracy:', mean(accuracy), 
      '\t Precision:', mean(precision), 
      '\t Recall:', mean(recall), 
      '\t F1:', mean(f1))




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)