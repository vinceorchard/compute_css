#https://maartengr.github.io/BERTopic/getting_started/representation/llm
import ast
import pandas as pd
import requests
import json
from tqdm import tqdm
tqdm.pandas()


#I) Upload datataset

df = pd.read_csv("data/datainput/temp/bertopic/topic_info.csv", index_col=0)


#II) Create prompts

# context prompt describes information given to all conversations


#III) Launch annotation

df["Representation"] = df["Representation"].apply(lambda x : ast.literal_eval(x))

model_name = "llama3.3"

#To do: add representative docs in prompt
def topic_llm_ollama(text_to_classify, model = model_name , url = "http://localhost:8000/api/generate"):
    context = """
    You are a helpful, respectful and honest assistant for labeling topics.
    """
    instruction = f"""
    A topic is described by the following keywords: {text_to_classify}.
    Based on the keywords about the topic, please create a short label of this topic. 
    Make sure you to only return the label and nothing more.
    """
    prompt = context + instruction
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False}
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return str(response.status_code)


#df = df.iloc[0:10]

df["topic_label"] = df["Representation"].progress_apply(lambda x : topic_llm_ollama(x))


df.to_csv("data/temp/bertopic/topic_info_labelled.csv")