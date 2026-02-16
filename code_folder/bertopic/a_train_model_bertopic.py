import time
import pandas as pd
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
tqdm.pandas()


#####################################
#********** LOADING DATA ***********#
#####################################

# Charger les discours

df = pd.read_csv("data/datainput/phrase_jt_2012_2024.csv", index_col = 0, sep = ';')

df_macro  = pd.read_csv("data/datainput/notices_fr2_2012_2024_emissions.csv")

df = df.merge(df_macro[["id","datdif"]], how = "left", left_on = "solr_id", right_on = "id")

df = df.drop(columns = ["id", "sentence_start", "sentence_end"])

del df_macro

######

df = df.dropna(subset = 'sentence_text')
df = df.reset_index(drop = True)
#df = df.iloc[100000:100500]

# Convertir les discours en liste
documents = df["sentence_text"].tolist()

del df

#####################################
#******** TRAIN BERTopic ***********#
#####################################

print(f"Training BERTopic model on {len(documents)} documents")

# Initialisation du modèle avec une réduction automatique des topics
vectorizer = CountVectorizer(stop_words=stopwords.words('french'))
topic_model = BERTopic(vectorizer_model=vectorizer)  # Réduction du nombre de topics à 9


# Chronométrer l'entraînement
start_time = time.time()

topics, probs = topic_model.fit_transform(documents)
elapsed_time = time.time() - start_time

print(f"BERTopic training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

topic_model.save("data/models/bert_topic_120226", serialization="pickle")
