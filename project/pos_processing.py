import pandas as pd
import re
import numpy as np


def get_clusters_consolidado(excel_de_lista_de_comentario_por_cluster):
    excel_clusters = pd.ExcelFile(excel_de_lista_de_comentario_por_cluster)

    comments_by_cluster = pd.DataFrame()

    for sheet in excel_clusters.sheet_names:
        aux_df = pd.read_excel(excel_de_lista_de_comentario_por_cluster,sheet_name=sheet)
        aux_df.rename(columns={'Cluster':'Cluster_value_hdb'}, inplace=True)
        aux_df['Cluster'] = sheet
        comments_by_cluster = comments_by_cluster.append(aux_df,ignore_index=True)

    # comments_by_cluster.drop(['Comment_vector','Content.1'],inplace=True)
    # print(comments_by_cluster)
    return comments_by_cluster[['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data',
       'Source','Cluster_value_hdb','Cluster']]



def get_lda_consolidado(excel_de_lista_de_lda_por_cluster):
    excel_lda = pd.ExcelFile(excel_de_lista_de_lda_por_cluster)

    lda_words_by_cluster = pd.DataFrame()
    lda_words_freq_by_cluster = pd.DataFrame()

    for sheet in excel_lda.sheet_names:
        is_word_freq = 'word_freq_'
        if is_word_freq in sheet:
            regex = r"Cluster_\d+"
            cluster = re.search(regex, sheet).group()

            aux_df = pd.read_excel(excel_de_lista_de_lda_por_cluster, sheet_name=sheet)
            aux_df['Cluster'] = cluster
            lda_words_freq_by_cluster = pd.concat([lda_words_freq_by_cluster,aux_df],ignore_index=True)
        else:

            aux_df = pd.read_excel(excel_de_lista_de_lda_por_cluster,sheet_name=sheet)
            aux_df['Cluster'] = sheet
            lda_words_by_cluster = pd.concat([lda_words_by_cluster,aux_df],ignore_index=True)

    lda_words_by_cluster.drop(['Unnamed: 0'],axis=1, inplace=True)
    lda_words_freq_by_cluster.drop(['Unnamed: 0'],axis=1, inplace=True)
    special_word = ['pra', "para", "e", "E", "", " ", "O", "A", "a", "o", 'bom', 'excelente', 'bem', 'agradavel',
                    'otimo', "Ótimo", 'boa', "ótimo", 'Bom', 'Boa', 'amazing', 'Amazing','otimas','super','\n']
    lda_words_by_cluster = lda_words_by_cluster[~lda_words_by_cluster['word'].isin(special_word) ]
    return lda_words_by_cluster,lda_words_freq_by_cluster

excel_de_lista_de_comentario_por_cluster = 'Clusters_Similarity_bar_DBSCAN_sem_janelamento_EPS3_Limpeza_MIN_SAMPLE_3.xlsx'
excel_de_lista_de_lda_por_cluster = 'LDA_Cluster_Similarity_bar_DBSCAN_sem_janelamento_EPS3_Limpeza_MIN_SAMPLE_3.xlsx'

comments_by_cluster = get_clusters_consolidado(excel_de_lista_de_comentario_por_cluster)
# Index(['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data',
#        'Source', 'Cluster'],r_Similarity_bar
lda_words_by_cluster,lda_words_freq_by_cluster = get_lda_consolidado(excel_de_lista_de_lda_por_cluster)
#Index(['word', 'score', 'weighted_score', 'freq', 'Cluster'], dtype='object')

df_cluster_word =lda_words_by_cluster.loc[lda_words_by_cluster.groupby('Cluster')['score'].idxmax()]
df_cluster_word.sort_values(by=['word'],inplace=True)

# df_cluster_word2 = lda_words_by_cluster.sort_values(['Cluster','score']).groupby('Cluster').tail(2)
# df_cluster_word2.sort_values(by=['word','Cluster','score'],inplace=True)

word_list = df_cluster_word.word.unique().tolist()

df_new_cluster_list = pd.DataFrame()
count = 1
for word in word_list:
    aux_df = df_cluster_word.loc[df_cluster_word['word']==word]
    aux_df['New_cluster'] = 'New_cluster_'+str(count)
    df_new_cluster_list=  df_new_cluster_list.append(aux_df,ignore_index=True)

    count+=1



comments_by_new_cluster = pd.merge(comments_by_cluster,df_new_cluster_list[['Cluster','word', 'New_cluster']], how='outer',on = 'Cluster')



for index_2, row_2 in df_new_cluster_list.iterrows():
    for index, row in comments_by_new_cluster.iterrows():
        if row['Cluster_value_hdb'] == -1:
            if row_2['word'] in row ['Content']:
                comments_by_new_cluster._set_value(index_2,'New_cluster',row_2['New_cluster'])
                comments_by_new_cluster._set_value(index_2, 'word', row_2['word'])
                comments_by_new_cluster._set_value(index_2, 'Cluster_value_hdb', -2)

# print(comments_by_new_cluster)
comments_by_cluster.fillna({'New_cluster':'Sem_new_cluster'}, inplace=True)
writer =pd.ExcelWriter('reclusterizacao_min_3.xlsx', engine = 'xlsxwriter')
new_cluster_list = comments_by_new_cluster.New_cluster.dropna().unique().tolist()

print(new_cluster_list)

for cluster in new_cluster_list:
    # print(cluster)
    aux_df_new_cluster = comments_by_new_cluster.loc[comments_by_new_cluster['New_cluster']==cluster]
    aux_df_new_cluster.to_excel(writer, sheet_name=cluster)
writer.save()





























