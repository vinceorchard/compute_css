import os
import time
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
tqdm.pandas()

#####################################
#********** LOADING DATA ***********#
#####################################

# Charger les discours
df = pd.read_csv("data/datainput/phrase_jt_2012_2024.csv", index_col = 0, sep = ';').reset_index().rename(
    columns = {"index" : "text_id"})

#df = df.iloc[0:500]

# Convertir les discours en liste
documents = df["sentence_text"].tolist()

df = df[["text_id"]]


#####################################
#****** SAVE RESULTS ***************#
#####################################
topic_model = BERTopic.load("data/models/bert_topic_120226_reducedTopics")

df = topic_model.get_document_info(documents, df = df)[["text_id", "Topic"]].rename(columns = {"Topic" : "bertopic"})

df["bertopic"].value_counts()

df.to_csv("data/temp/bertopic_unlabelled.csv")

topic_model.get_topic_info().to_csv("data/temp/topic_info.csv")
