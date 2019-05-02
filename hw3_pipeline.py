'''
Machine Learning Pipeline for Homework 3
'''

import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from datetime import datetime
from sklearn.metrics import *
import matplotlib.pyplot as plt
'''
Import the data in csv file
'''
def read_data(filename):
	'''
	Convert csv file to a pandas dataframe
	Input: filename (str)
	'''
	data = pd.read_csv(filename)

	return data

'''
Convert datatypes
'''
def convert_type(data, colname, target_type):
	'''
	Convert datatype of a column
	Input: data (dataframe), colname (str), type (str)
	'''
	data[colname] = data[colname].astype(target_type)

'''
Check if the column contains NULLs
'''
def if_null(data, colname):
	'''
	Input: data(dataframe)，colname(str)
	'''
	return data[colname].isnull().values.any()

'''
Find the distribution of a variable in the dataset
'''
def describe_data(data, colname):
	'''
	Input: data(dataframe), colname(str)
	'''
	return data[colname].describe()

'''
Make the boxplot of a variable in the dataset
'''
def boxplot(data, colname):
	'''
	Input: data(dataframe), colname(str)
	'''
	return data[colname].plot.box()

'''
Make the density plot of a variable in the dataset
'''
def density_plot(data, colname):
	'''
	Input: data(dataframe), colname(str)
	'''
	return data[colname].plot.density()

'''
Find summaries of all variables that we are interested in
'''
def find_summaries(data, colnames):
	'''
	Input: data(dataframe), colnames (list)
	'''
	return data[colnames].describe()

'''
Find correlations between variables
'''
def find_corr(data, col1, col2):
	'''
	Input: data(dataframe). col1(str), col2(str)
	'''
	return data[col1].corr(data[col2])

'''
Discretize a set of columns in a dataset
'''
def discretize_col(data, columns):
    '''
    To discretize the continuous variable into three discrete variables: 0, 1, and 2;
    the boundaries are the minimum value, the 25% quantile, the 75% quantile, and the maximum value.
    
    Inputs: data, pandas dataframe
            columns, list
    '''
    for column in columns:
    	data[column] = pd.cut(data[column], bins=[data[column].min(), data[column].quantile(0.25), data[column].quantile(0.75),
                                              	data[column].max()], labels=[0,1,2], include_lowest=True)
'''
Fill in NA values with mean
'''
def fill_na(data, columns):
	'''
	Input: data, pandas dataframe
	       columns, list
	'''
	for column in columns:
		if data[column].isnull().any():
			data[column] = data[column].fillna(data[column].median())

'''
Convert the label to dummy variables
'''
def label_to_dummy(item, bar):
    '''
    item: int
    bar: int
    '''
    if item >= bar:
        result = 1
    else:
        result = 0
    return result

'''
Convert columns in categorical variables to dummy variables
'''
def to_dummy(data, column):
    '''
    data, pandas dataframe
    column, list
    '''
    data = pd.get_dummies(data, columns=column)
    
    return data

'''
Slice time series data by time
'''
def slice_time_data(data, start_time, end_time):
	'''
	data, pandas dataframe
	start_time, string
	end_time, string
	'''
    return data[(data['date_posted'] >= start_time) & (data['date_posted'] <= end_time)]


'''
The following codes are referenced from the folloiwng website:
https://github.com/rayidghani/magicloops/blob/master/simpleloop.py, credit to Rayid Ghani
'''
def define_clfs_params():
	'''
	Create the dictionary that links models name to the models objects
	'''

    models = {'BG_N_Estimators_10': BaggingClassifier(n_estimators=10),
              'BG_N_Estimators_100': BaggingClassifier(n_estimators=100),
              'RF_N_Estimators_10': RandomForestClassifier(n_estimators=10, n_jobs=-1),
              'RF_N_Estimators_100': RandomForestClassifier(n_estimators=100, n_jobs=-1),
              'LR_l1': LogisticRegression(penalty='l1', C=1),
              'LR_l2': LogisticRegression(penalty='l2', C=1),
              'SVM_C_l1': svm.LinearSVC(random_state=0, penalty='l1', dual=False),
              'SVM_C_l2': svm.LinearSVC(random_state=0, penalty='l2'),
              'GB_N_Estimators_10': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
              'GB_N_Estimators_100': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100),
              'DT_GINI': DecisionTreeClassifier(criterion='gini'),
              'DT_ENTROPY': DecisionTreeClassifier(criterion='entropy'),
              'KNN_N_10': KNeighborsClassifier(n_neighbors=10),
              'KNN_N_100': KNeighborsClassifier(n_neighbors=100)}
    
    return models


def joint_sort_descending(l1, l2):
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, preds_at_k)
    return recall


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.title(model_name)
    plt.show()


def clf_loop(models, models_to_run, X_train, X_test, y_train, y_test):
    results_df = pd.DataFrame(columns=('model_type', 'accuracy', 'f1_score', 'auc-roc','p_at_5',
                                        'p_at_10', 'p_at_20', 'r_at_5', 'r_at_10', 'r_at_20'))
    for model_name in models_to_run:
        model = models[model_name]
        if 'SVM' in model_name:
            y_pred_probs = model.fit(X_train, y_train).decision_function(X_test)
        else:
            y_pred_probs = model.fit(X_train, y_train).predict_proba(X_test)[:,1]
        y_pred = model.fit(X_train, y_train).predict(X_test)
        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
        results_df.loc[len(results_df)] = [model_name, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred),
                                           roc_auc_score(y_test, y_pred_probs),
                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                           precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                           recall_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                           recall_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                           recall_at_k(y_test_sorted, y_pred_probs_sorted, 20.0)]
        plot_precision_recall_n(y_test, y_pred_probs, model_name)
    return results_df