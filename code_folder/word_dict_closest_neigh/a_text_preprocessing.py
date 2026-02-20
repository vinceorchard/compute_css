import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
tqdm.pandas()
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
from itertools import chain
import re

nlp = spacy.load('fr_core_news_lg')

# Charger les discours
df = pd.read_csv("data/datainput/phrase_jt_2012_2024.csv", index_col = 0, sep = ';')

df["sentence_text"] = df["sentence_text"].progress_apply(lambda x : re.sub("  ", " ", x))
df['sentence_text'] = df['sentence_text'].str.lower()
df['sentence_text'] = df['sentence_text'].progress_apply(lambda x: unidecode(x.lower()))
df['sentences'] = df['sentence_text'].progress_apply(lambda x: sent_tokenize(x))


df = df['sentences']

corpus = list(chain.from_iterable(df))

# Create a DocBin object to store the documents
doc_bin = DocBin()

# Process the corpus and add to DocBin
for text in tqdm(corpus):
    doc = nlp(text)
    doc_bin.add(doc)

# Save the DocBin to disk
doc_bin.to_disk("data/models/corpus.spacy")
