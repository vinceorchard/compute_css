import pandas as pd
import gensim
from spacy.lang.en.stop_words import STOP_WORDS as fr_stop
import pickle as pk
from wordcloud import WordCloud
from tqdm import tqdm
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("fr_core_news_lg")
tqdm.pandas()


df = pd.read_csv("data/dataoutput/extract_unemployed_mention_label.csv", index_col = 0)

df_all = pd.read_csv("data/datainput/phrase_jt_2012_2024.csv", index_col = 0, sep = ';')

################################
################################

def quick_cleaning(text):
    tokens = gensim.utils.simple_preprocess(text, deacc=True, min_len=3)
    tokens = [t for t in tokens if t not in fr_stop]
    return " ".join(tokens)

texts = df["sentence_text"].progress_apply(quick_cleaning).tolist()
docs = list(nlp.pipe(texts, batch_size=1000, n_process=4))
df["sentence_text_clean"] = docs

texts = df_all["sentence_text"].progress_apply(quick_cleaning).tolist()
docs = list(nlp.pipe(texts, batch_size=1000, n_process=4))
df_all["sentence_text_clean"] = docs


##############

##############################
lemmas_all = {}

for text in tqdm(df_all["sentence_text_clean"]):
    token = str(token)
    for token in text:
        #lemma = token.lemma_
        #lemma = token
        if token in lemmas_all:
            lemmas_all[token] += 1
        else:
            lemmas_all[token] = 1


with open('data/dataoutput/words_frequency_Fdataset.pk', 'wb') as f:
    pk.dump(lemmas_all, f)


##############################
lemmas_chomeurs = {}

for text in tqdm(df["sentence_text_clean"]):
    token = str(token)
    for token in text:
        #lemma = token.lemma_
        #lemma = token
        if token in lemmas_chomeurs:
            lemmas_chomeurs[token] += 1
        else:
            lemmas_chomeurs[token] = 1


with open('data/dataoutput/words_frequency_chomeur_dataset.pk', 'wb') as f:
    pk.dump(lemmas_chomeurs, f)


#
# adjectives = {}
#
# for text in tqdm(df["sentence_text_clean"]):
#     for token in text:
#         if token.pos_ == 'ADJ':
#             #lemma = token.lemma_
#             lemma = token
#             if lemma in adjectives:
#                 adjectives[str(lemma)] += 1
#             else:
#                 adjectives[str(lemma)] = 1
#

#with open('09_graphs/gendered_communication/adjective_frequency_Fdataset.pk', 'wb') as f:
#    pk.dump(adjectives, f)


# wc = WordCloud(background_color="white", width=1000, height=1000, max_words=30, relative_scaling=0.5,
#                    normalize_plurals=False).generate_from_frequencies(lemmas_all)
#
# plt.figure( figsize=(30,30))
# plt.imshow(wc)
# plt.axis("off")
# plt.savefig("graphs/wordcloud_unemployed.pdf")
# plt.close()
