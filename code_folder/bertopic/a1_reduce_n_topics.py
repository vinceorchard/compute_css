import os
import time
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
tqdm.pandas()

topic_model = BERTopic.load("data/models/bert_topic_120226")

df = pd.read_csv("data/datainput/phrase_jt_2012_2024.csv", index_col = 0, sep = ';').reset_index().rename(
    columns = {"index" : "text_id"})

documents = df["sentence_text"].tolist()

# Chronométrer l'entraînement
start_time = time.time()

# Further reduce topics
topic_model.reduce_topics(documents, nr_topics=80)

elapsed_time = time.time() - start_time

print(f"N of topics of BERTopic reduced in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

topic_model.save("data/models/bert_topic_120226_reducedTopics", serialization="pickle")
