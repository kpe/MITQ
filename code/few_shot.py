import openai
import pandas as pd
import time
import backoff
import tiktoken
import csv 
import math
from grade import num_tokens_from_messages
import os

openai.api_key = os.getenv('OpenAI_API_Key')
def few_shot_response(system, question, fs_qs, cot=False, max_tokens=8192):
    #course_num = adjusted_course_num(course)
    #similar_qs = get_n_similar_questions(course, number, n)
    messages = []
    messages.append({"role":"system", "content":f"You are {system} course.\n"
                                          f"Your task is to answer the following question given the solutions to {len(fs_qs)} similar questions."})
    content = f"Here are {len(fs_qs)} similar questions and solutions \n"
    #for i in range(n):
    for i in range(len(fs_qs)):
        #sim_course, sim_num = similar_qs[i]
        #q_info = dict(class_dataframes[sim_course].loc[int(sim_num)])
        #print(q_info)
        #content += f"Question #{i+1}:\n {q_info['Question']} \n Solution: \n {q_info['Solution']} \n"
        content += f"Question #{i+1}:\n {fs_qs[i][0]} \n Solution: \n {fs_qs[i][1]} \n"
        #print(i, num_tokens_from_messages([{'i':f"Question #{i+1}:\n {q_info['Question']} \n Solution: \n {q_info['Solution']} \n"}]))
        content += f"What is the answer to the following question?:\n {question}"
        if cot:
            content += "Let's think step by step."

    messages.append({"role":"user", "content":content})
    #print(content)
    try:
        completion = openai.ChatCompletion.create(
            model=os.getenv('Prompt_Engine'),
            temperature=0,
            max_tokens=max_tokens - num_tokens_from_messages(messages),
            messages=messages)
        return completion["choices"][0]["message"]["content"]
        #return num_tokens_from_messages(messages)
    except openai.error.APIError as e:        
        time.sleep(45)
        return few_shot_response(system, question, fs_qs, cot)
    except openai.error.APIConnectionError as e:
        time.sleep(45)
        return few_shot_response(system, question, fs_qs, cot)
    except openai.error.RateLimitError as e:
        time.sleep(45)
        return few_shot_response(system, question, fs_qs, cot)
