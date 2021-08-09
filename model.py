# model file for you to edit
import pandas as pd
import pandas as pd1
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import train_test_split
global X_train, y_train, X, y, X_test, y_test
import logging

logging.basicConfig(filename="logfilename.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    filemode='w',
                    force = True)

df1 = pd1.read_csv('Config.csv', sep=',')
print(df1)
print(df1["DATA_PATH"])
SplitRatio=df1["SplitRatio"]
limit=df1["CostLimit"]
CV=df1["CV"]
print(SplitRatio) #these have 0 in fornt when reading from CSV, dont know why
print(limit)
print(CV)
#so ended up reassigning these variables to correct numbers
SplitRatio=0.8
limit=12000
CV=5
print(SplitRatio)
print(limit)
print(CV)

def read_logfile(log_filepath):
    with open(log_filepath) as f:
        for line in f:
            print(line)

# Created a data processing function

def data_processing(filepath, SplitRatio, limit):
    """This is the function to select the relevent data and splitting data into test and train"""
    try:
        print("INFO: File path exists")
        df = pd.read_csv(filepath, sep=',')
        # Shuffle the datarows randomly, to be sure that the ordering of rows is somewhat random:
        df = df.sample(frac=1)
        # Use the method "drop nan" on the DataFrame itself to force it to remove rows with null/nan-values:
        df = df.dropna(axis=0, how='any')
        print('Price limit set for classification: ${}'.format(limit))
        # Select the columns/features from the Pandas dataframe that we want to use in the model:
        X = np.array(df[['Age', 'KM']])  # we only take the first two features.
        y = 1*np.array(df['Price']>limit)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SplitRatio)
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("WARNING: File path doesn't exist")
    except ValueError:
        print("WARNING: Value not interger")


# Create a linear regression model that we can train:
# This function is to train the model, with training data and CV as input to the fuction and it returns the traing results
def model_training(X_train, y_train, CV): 
    """This function is to train the model, with training data and CV as input to the fuction and it returns the traing results"""
    clf = tree.DecisionTreeClassifier(max_depth=3)
    #clf.fit(X_train, y_train)
    # Train the model using CV and multiple scoring on the data we have prepared:
    cv_results = cross_validate(clf, # Provide our model to the CV-function
                            X_train, # Provide all the features (in real life only the training-data)
                            y_train, # Provide all the "correct answers" (in real life only the training-data)
                            scoring=('f1', 'precision', 'recall', 'accuracy'), 
                            cv=CV # Cross-validate using 5-fold (K-Fold method) cross-validation splits
                           )
    return cv_results


# Print some information about the linear model and its parameters:

#this function is to print the results of the training model
def print_results(cv_results):
    """This function is to print the results of the training model"""
    #print(clf)
    F1_pos   = cv_results['test_f1']
    P_pos   = cv_results['test_precision']
    R_pos   = cv_results['test_recall']
    A   = cv_results['test_accuracy']
    print('\n-------------- Scores ---------------')
    print('Average F1:\t {:.2f} (+/- {:.2f})'.format(F1_pos.mean(), F1_pos.std()))
    print('Average Precision (y positive):\t {:.2f} (+/- {:.2f})'.format(P_pos.mean(), P_pos.std()))
    print('Average Recall (y positive):\t {:.2f} (+/- {:.2f})'.format(R_pos.mean(), R_pos.std()))
    print('Average Accuracy:\t {:.2f} (+/- {:.2f})'.format(A.mean(), A.std()))

file_path = "Dataset/Toyota.csv"
#if Path(file_path).exists():  #doesnt work
print("INFO: File path exists")
X_train, X_test, y_train, y_test = data_processing(file_path, SplitRatio, limit)
cv_results = model_training(X_train, y_train, CV)
print_results(cv_results)
#else:
#print("WARNING: File path doesn't exist")

read_logfile("logfilename.log")
            
def pipeline():
    #file_path = np1.array(df1[['DATA_PATH']])
    X_train, X_test, y_train, y_test = data_processing(file_path, SplitRatio, limit)
    cv_results = model_training(X_train, y_train, CV)
    print_results(cv_results)
