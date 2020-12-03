# Project library import
# import project.import_data as import_data
# import project.clusterize as clusterize
# import project.word_ebbending as word_ebbending
# import project.output_analysis as output_analysis
# import project.LDA as LDA
import import_data
import clusterize
import word_ebbending
import output_analysis
import LDA

# External library import
import pandas as pd
import numpy as np
		
# Main();
print('Parameters: \n')
# data = import_data.import_data_bar() # dataset do bar sem quebra
data = import_data.new_data(import_data.import_data_bar()) # dataset do bar com quebra

# data = import_data.import_data_edc() # dataset do EDC sem quebra
# data = import_data.new_data(import_data.import_data_edc()) # dataset do EDC com quebra
data_v2 = word_ebbending.get_comment_vector(data)
print('data_v2: \n')
print(data_v2)
sim_matrix = word_ebbending.similarity_matrix(data_v2)
df_pool, data_v3, clusters, rating, weight = clusterize.dbscan_clustering(sim_matrix, data_v2, False)

# mudar aqui os nomes dos inputs e outputs
excel_input = 'DBSCAN_EPS3_Limpeza_MIN_SAMPLE_3_So_E'+'.xlsx'
excel_output = 'LDADBSCAN_EPS3_Limpeza_MIN_SAMPLE_3_So_E'+'.xlsx'
n_topic = 1
writer =pd.ExcelWriter(excel_input, engine = 'xlsxwriter')
count = 0
print('L DF POOL:', len(df_pool))
for cluster_segment in df_pool:
	print(count,'\n')
	print(cluster_segment,'\n')
	cluster_segment.to_excel(writer, sheet_name='Cluster_'+str(count))
	count += 1
writer.save()

LDA.lda_from_clusterized_excel(n_topic,excel_input, excel_output)
# output_analysis.bubble_chart(rating, clusters, weight)
