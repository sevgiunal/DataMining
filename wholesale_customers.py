import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt




# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	data = pd.read_csv(data_file)
	data = data.drop(columns=['Channel', 'Region'])
	return data

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	summary = df.describe().T[['mean', 'std', 'min', 'max']]
	summary['mean'] = summary['mean'].round(0).astype(int)
	summary['std'] = summary['std'].round(0).astype(int)
	return summary

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	means = df.mean()
	stds = df.std().replace(0, 1) # Replace 0 std with 1 to avoid division errors
	df_standardized = (df - means) / stds
	return df_standardized

# Given a dataframe df and a number 
# of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
	kmeans.fit(df)
	return pd.Series(kmeans.labels_, index=df.index, name="Cluster")

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    # Initialize KMeans with KMeans++ initialization
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)

    # Fit KMeans model
    kmeans.fit(df)

    # Return cluster labels as a Pandas Series
    return pd.Series(kmeans.labels_, index=df.index, name="Cluster")

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	cluster = AgglomerativeClustering(n_clusters=k)
	cluster_labels = cluster.fit_predict(df)

	return pd.Series(cluster_labels, index = df.index, name = "Cluster")

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	return silhouette_score(X,y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	results = []
	df_standardized = standardize(df)
	k_clusters = [3,5,10]

	for data_type, dataset in [("Original", df), ("Standardized", df_standardized)]:
		for k in k_clusters:
			for i in range(10):
				y_kmeans = kmeans(dataset, k)
				silhouette = clustering_score(dataset, y_kmeans)  # Compute Silhouette Score
				
				results.append({
                	"Algorithm": "KMeans",
                	"Data Type": data_type,
                	"k": k,
                	"Silhouette Score": silhouette
            	})

				# Run Agglomerative Clustering (only once per k)
				y_agglomerative = agglomerative(dataset, k)
				silhouette = clustering_score(dataset, y_agglomerative)

				results.append({
					"Algorithm": "Agglomerative",
					"Data Type": data_type,
					"k": k,
					"Silhouette Score": silhouette
				})

	return pd.DataFrame(results)

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf.loc[rdf['Silhouette Score'].idxmax()]

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	columns = df.columns.tolist()
	num_attributes = len(columns)


	df_standardized = standardize(df)  # Standardize the dataset
	y_clusters = kmeans(df_standardized, k=3)  # Get cluster assignments for k=3

	plt.figure(figsize=(12, 15))  # Slightly increase figure size for better spacing


	plot_num = 1

	for i in range (num_attributes):
		for j in range(i+1, num_attributes):
			plt.subplot(5,3,plot_num)
			plt.scatter(df_standardized[columns[i]], df_standardized[columns[j]], c=y_clusters, cmap='tab20', alpha=0.7)
			plt.xlabel(columns[i])
			plt.ylabel(columns[j])
			plt.title(f'{columns[i]} vs {columns[j]}')
			plot_num += 1
	
	plt.tight_layout()
	plt.savefig("scatter_plots.png", dpi=300)  # Save in the local folder
	plt.close()  # Close to free memory

	pass
