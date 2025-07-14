# Part 3: Text mining.
import pandas as pd
from collections import Counter
import urllib.request
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
import numpy as np



# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the coronavirus_tweets.csv file.
def read_csv_3(data_file):
	data = pd.read_csv(data_file, encoding='latin-1')
	return data

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    return df['Sentiment'].unique().tolist()


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	count = df['Sentiment'].value_counts()
	return count.index[1] 

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):

	extremely_positive = df[df['Sentiment'] == "Extremely Positive"]

	date_counts = extremely_positive['TweetAt'].value_counts()
	return date_counts.idxmax()

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ', regex=True).str.strip()
	
	return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda tweet: tweet.split())

	return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	total_words = sum(len(tweet) for tweet in tdf['OriginalTweet'])
	return total_words

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	unique_words= set(word for tweet in tdf['OriginalTweet'] for word in tweet)
	return len(unique_words)

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	all_words = [word for tweet in tdf['OriginalTweet'] for word in tweet]
	word_counts = Counter(all_words)
	common_words = [word for word, count in word_counts.most_common(k)]
	return common_words

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stop_words = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
	with urllib.request.urlopen(stop_words) as response:
		stop_words = set(response.read().decode('utf-8').splitlines())

	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
        lambda tweet: [word for word in tweet if word not in stop_words and len(word) > 2]
    )

	return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    ps = PorterStemmer()
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
        lambda tweet: [ps.stem(word) for word in tweet]  # Apply stemming to each word
    )
    return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.9, ngram_range=(1, 2))


	X = vectorizer.fit_transform(df['OriginalTweet'].values)
	y = df['Sentiment'].values

	# Train Multinomial Naive Bayes model
	model = MultinomialNB()
	model.fit(X, y)

	# Predict on the training set
	y_pred = model.predict(X)

	return np.array(y_pred)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	return round(accuracy_score(y_true, y_pred), 3)
