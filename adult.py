
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	data = pd.read_csv(data_file)
	data = data.drop(columns=['fnlwgt'])

	return data

# Return the number of rows (instances) in the pandas dataframe df.
def num_rows(df):
	num_of_rows = df.shape[0]
	return num_of_rows

# Return a list with the column names (attributes) in the pandas dataframe df.
def column_names(df):
	column_names = df.columns.tolist()
	return column_names

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	missing = df.isnull().sum().sum()
	return missing

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	columns_with_missing = df.columns[df.isnull().any()].tolist()
	return columns_with_missing

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	filter = df[(df['education'] == 'Bachelors') | (df['education'] == 'Masters')]
	percentage = (filter.shape[0] / df.shape[0]) * 100
	return round(percentage, 1)

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df_new = df.dropna()
	return df_new

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	df = df.drop(columns = ['class'])
	return pd.get_dummies(df, drop_first=True, dtype=int)

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	label_encoder = preprocessing.LabelEncoder()
	return pd.Series(label_encoder.fit_transform(df['class']), index=df.index, name="income")


# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	dt = DecisionTreeClassifier()
	dt.fit(X_train, y_train)

	y_pred = dt.predict(X_train)
    
	return pd.Series(y_pred, index=X_train.index, name="Predicted")

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	
	accuracy = accuracy_score(y_true, y_pred)  # Compute accuracy

	return 1 - accuracy

