import pandas as pd
import requests
import json
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


###########################

#I) Import data

df = pd.read_csv("data/dataoutput/extended_keywords_extraction.csv", index_col =0)

df = df[["sentence_text","solr_id", "sentence_id"]]

###########################
#II) define variables and functions

model_name = "llama4:latest"


instruction_filtering_employment_policy = """
ROLE : Tu es un expert en analyse du discours médiatique français. 
MISSION: Détermine si cet extrait de presse audiovisuelle mentionne le chômage, les chômeurs ou les demandeurs d'emploi.
CRITÈRES DE CLASSIFICATION : 
    - OUI : mention claire, directe ou indirecte
    - VERIFICATION : mention possible du chômage ou des chômeurs mais sans certitude suffisante
    - NON : aucune référence même indirecte
FORMAT DE RÉPONSE: OUI | NON | VERIFICATION (uniquement, sans explications)
"""


session = requests.Session()

def topic_llm_ollama(text_to_classify, model = model_name , url = "http://localhost:8000/api/generate"):
    instruction = instruction_filtering_employment_policy
    prompt = instruction + text_to_classify
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False}
    headers = {
        "Content-Type": "application/json"
    }
    response = session.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return str(response.status_code)


def classify_all(texts, max_workers=4):
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(topic_llm_ollama, t): i for i, t in enumerate(texts)}

        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

    return results


#temp = df.iloc[0:80].copy()
#temp["result"] = classify_all(temp["sentence_text"].tolist())

#df["result"] = classify_all(df["sentence_text"].tolist())
#df.to_csv("data/dataouput/unemployed_llm_filtered.csv")
#df["llm_output"] = df["sentence_text"].progress_apply(lambda x : topic_llm_ollama(x))


def apply_in_batches(df, col_input, col_output, f, other_col_to_save, n_batches,
                     file_name = "topics_prompt_"+model_name, path_to_store_bashes = "data/temp/bashes/"):
    batch_size = int(np.ceil(len(df) / n_batches))
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        df_temp = df[[col_input] + other_col_to_save].iloc[start_idx:end_idx]
        df_temp[col_output] = None
        df_temp[col_output] = f(df_temp[col_input].tolist())
        df_temp[[col_output] + other_col_to_save].to_csv(path_to_store_bashes + file_name + "_bash_" + str(i + 1) + ".csv")

###########################
#III) Lancer annotation

#df = df.iloc[0:20]
#df["topic_llm_" + model_name.replace(".","_")] = df["text"].progress_apply(lambda x : topic_llm_ollama(x))

# Chronométrer l'entraînement
print(f"Start LLM annotation")

start_time = time.time()

apply_in_batches(df, col_input = "sentence_text", col_output = "mention_chomeurs", f = classify_all, other_col_to_save = ["solr_id", "sentence_id"],
                 n_batches = 10, file_name = "mention_chomeurs", path_to_store_bashes = 'data/temp/bash/')

elapsed_time = time.time() - start_time

print(f"LLM annotation completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
