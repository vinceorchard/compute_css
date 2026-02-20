import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
tqdm.pandas()
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
from itertools import chain

nlp = spacy.load('fr_core_news_lg')

# Charger les discours
df = pd.read_csv("data/temp/parsing_full_corpus.csv", encoding="utf8")


df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: unidecode(x.lower()))
df['sentences'] = df['text'].progress_apply(lambda x: sent_tokenize(x))


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
