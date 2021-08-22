# model file for you to edit
import pandas as pd
import pandas as pd1
import numpy as np
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
global X_train, y_train, X, y, X_test, y_test
import logging
import click
import pickle


logging.basicConfig(filename="logfilename.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    filemode='w',
                    force = True)

def read_logfile(log_filepath: str):
    with open(log_filepath) as f:
        for line in f:
            print(line)

# Created a data processing function

def data_processing(filepath: str, SplitRatio: float, limit: float):
    """This is the function to select the relevent data and splitting data into test and train"""
    try:
        logging.info("INFO: File path exists")
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
        logging.info("WARNING: File path doesn't exist")
    except ValueError:
        logging.info("WARNING: Value not interger")


# Create a linear regression model that we can train:
# This function is to train the model, with training data and CV as input to the fuction and it returns the traing results
def model_develop(X_train, y_train): 
    """This function is to train the model, with training data and returns the model"""
    mdl=LinearRegression()
    # Train the model using the data we have prepared:
    mdl.fit(X_train, y_train)
    return mdl


# this function is savemodel to pickle
def savepicklemdl(model):
    with open ("arun_model.pkl", "wb") as my_stream:
        pickle.dump(model, my_stream)


@click.command()
@click.option('--limit', default=None)

def model(limit):
    df1 = pd1.read_csv('Config.csv', sep=',')
    #printing configuration file
    logging.info(df1)
    #assigning variables
    file_path =df1["DATA_PATH"][0]
    SplitRatio=df1["SplitRatio"][0] 
    if limit is None:
        limit=df1["CostLimit"][0]
    limit=int(limit)
    CV=df1["CV"][0]   
    if Path(file_path).exists():
        logging.info('INFO: File path exists')
        X_train, X_test, y_train, y_test = data_processing(file_path, SplitRatio, limit)
        modeldeveloped = model_develop(X_train, y_train)
        logging.info('Model Trained')
        savepicklemdl(modeldeveloped)
        logging.info('Model Exported')
    else:
        logging.info('WARNING: File path does not exist')
    read_logfile("logfilename.log")

if __name__ == '__main__':
    logging.info('Welcome to Machine LEarning')
    model()
