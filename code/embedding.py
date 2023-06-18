import openai
import json
from sentence_transformers import util
import os
import pandas as pd

openai.api_key = os.getenv('OpenAI_API_Key')

def load_questions(file):
    questions = []
    sheet = pd.read_csv(file)
    sheet = sheet.fillna('null')
    for i in range(len(sheet['Question'])):
        if sheet.loc[i, "Question Number"] == 'null':
            break
        question = sheet.loc[i, 'Question']
        questions.append(question)
    return questions

def make_embeddings(file):
    list_of_embeddings = []
    print("Currently embedding " + file + "...")
    sheet = pd.read_csv(file)
    sheet = sheet.fillna('null')
    for i in range(len(sheet['Question'])):
        print(i)
        # a null(empty entry) in question is treated as cutoff
        if sheet.loc[i, "Question Number"] == 'null':
            break
        if (sheet.loc[i, "Question Type"].lower() == "image"):
            continue
        raw_question = sheet.loc[i, 'Question']
        embedding = openai.Embedding.create(input=raw_question, model=os.getenv('Embedding_Engine'))['data'][0]['embedding']
        list_of_embeddings.append(embedding)
    embeddings = {'list_of_embeddings': list_of_embeddings}
    with open(file + '_embeddings.json', 'w') as f:
        f.write(json.dumps(embeddings))


def get_embeddings(file): # from file, with dimensions (n x d)
    with open(file, 'r') as f:
        points = json.load(f)['list_of_embeddings']
    return points


def get_most_similar(embeddings, target_embedding): # questions in embedding space by cosine similarity.
    cos_sims = []
    cos_to_num = {}
    for j in range(len(embeddings)):
        cos_sim = util.cos_sim(target_embedding, embeddings[j]).item()
        cos_to_num[cos_sim] = j
        cos_sims.append(cos_sim)
    ordered = sorted(cos_sims, reverse=True)
    closest_qs = [cos_to_num[val]+1 for val in ordered]
    return closest_qs
