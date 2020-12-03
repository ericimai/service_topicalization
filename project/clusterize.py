# Project library
# import project.word_ebbending as word_ebbending
# import project.import_data as import_data
import word_ebbending
import import_data

# External library
from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets.samples_generator import make_blobs
from yellowbrick.cluster import KElbowVisualizer
# from pyspark.ml.clustering import BisectingKMeans
# from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import hdbscan


# Elbow method:
def elbow_method(bag_of_words):
	model = KMeans()
	visualizer = KElbowVisualizer(model,k = (2,12))
	visualizer.fit(bag_of_words)
	plt.cla()
	plt.close()
	return visualizer.elbow_value_

def clusterize(bag_of_words):
	optimal_k = elbow_method(bag_of_words)
	kmeans = KMeans(n_clusters=41)
	return kmeans, optimal_k

def clusterize_structure(bag_of_words):
	df_bw = pd.DataFrame(bag_of_words)
	kmeans, optimal_k = clusterize(bag_of_words)
	predict = kmeans.fit_predict(df_bw.values)
	df_bw["clusters"] = kmeans.fit_predict(df_bw.values)
	df_bw.groupby("clusters").aggregate("mean").plot.bar()
	plt.show()

def rating_range(data):
	max_rating = max(data["Rating"])
	min_rating = min(data["Rating"])
	pace = (max_rating - min_rating)/3
	rating_range = [min_rating, min_rating + pace, min_rating + 2*pace, max_rating]

	return rating_range

def clusterize_share(bag_of_words, data, window):
	# bag_of_words: saída do similarity_matrix
	# data: saída do import_data()
	df_pool = []
	clusters = []
	rating = []
	weight = []

	index = data.index.values
	df_bw = pd.DataFrame(bag_of_words)
	kmeans, optimal_k = clusterize(bag_of_words)
	predict = kmeans.fit_predict(df_bw.values)
	data["Cluster"] = predict

	if(window == True):
		rating_limits = rating_range(optimal_k)
		for cluster in range(90):
			clusters.append(cluster)
			data_segment = data[(data['Cluster'] == cluster) & (data['Rating'] >= rating_limits[0]) & (data['Rating'] < rating_limits[1])]
			rating.append(1)
			weight.append(len(data_segment.index))
			df_pool.append(data_segment)

			clusters.append(cluster)
			data_segment = data[(data['Cluster'] == cluster) & (data['Rating'] >= rating_limits[1]) & (data['Rating'] < rating_limits[2])]
			rating.append(2)
			weight.append(len(data_segment.index))
			df_pool.append(data_segment)

			clusters.append(cluster)
			data_segment = data[(data['Cluster'] == cluster) & (data['Rating'] >= rating_limits[2])]
			rating.append(3)
			weight.append(len(data_segment.index))
			df_pool.append(data_segment)
	else:
		for cluster in range(41):
			data_segment = data[(data['Cluster'] == cluster)]
			df_pool.append(data_segment)

	return(df_pool, data, clusters, rating, weight)

def dbscan_clustering(bag_of_words, data, window):
	df_pool = []
	clusters = []
	rating = []
	weight = []

	index = data.index.values			
	# df_bw = pd.DataFrame(bag_of_words)
	predict = DBSCAN(eps=3,min_samples=3).fit_predict(bag_of_words)
	# predict = DBSCAN(eps=3).fit_predict(bag_of_words)
	print("PREDICT\n")
	print(predict,'\n')
	data["Cluster"] = predict

	if(window == True):
		rating_limits = rating_range(data)
		for cluster in range(-1,max(predict)):
			clusters.append(cluster)
			data_segment = data[(data['Cluster'] == cluster) & (data['Rating'] >= rating_limits[0]) & (data['Rating'] < rating_limits[1])]
			rating.append(1)
			weight.append(len(data_segment.index))
			df_pool.append(data_segment)

			clusters.append(cluster)
			data_segment = data[(data['Cluster'] == cluster) & (data['Rating'] >= rating_limits[1]) & (data['Rating'] < rating_limits[2])]
			rating.append(2)
			weight.append(len(data_segment.index))
			df_pool.append(data_segment)

			clusters.append(cluster)
			data_segment = data[(data['Cluster'] == cluster) & (data['Rating'] >= rating_limits[2])]
			rating.append(3)
			weight.append(len(data_segment.index))
			df_pool.append(data_segment)
	else:
		for cluster in range(-1,max(predict)):
			data_segment = data[(data['Cluster'] == cluster)]
			df_pool.append(data_segment)	

	return(df_pool, data, clusters, rating, weight)
