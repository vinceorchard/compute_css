import pandas as pd


topic_info_labelled = pd.read_csv("data/datainput/temp/bertopic/topic_info_labelled.csv")

topic_info_labelled = topic_info_labelled[["Topic", "topic_label"]].rename(columns = {"Topic" : "bertopic"})

df = pd.read_csv("data/datainput/temp/bertopic/bertopic_unlabelled.csv",index_col=0)

df = df.merge(topic_info_labelled, how = "left", on = "bertopic")

df.to_csv("data/temp/bertopic/bertopic_labelled.csv")


##########
# Open dataframe
df_text = pd.read_csv("data/datainput/temp/parsing_full_corpus.csv", encoding="utf8")

df = df[df["bertopic"].isin([20,24, 25,27])].reset_index(drop=True)

df.merge(df_text, how = "left", left_on="text_id", right_index=True).to_csv("data/dataoutput/filtered_datasets/bertopic_extraction_labourmarket.csv")


df = df[df["bertopic"]==25].reset_index(drop=True)

df.merge(df_text, how = "left", left_on="text_id", right_index=True).to_csv("data/dataoutput/filtered_datasets/bertopic_extraction_unemployment.csv")

