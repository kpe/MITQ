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
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def self_critique_response(system, course_name, question, prev_answer, critiques, max_tokens=8192):
    messages = []
    messages.append({"role":"system", "content":f"You are {system} course.\n"
                                          f"Your task is to answer the following question given a previous answer to the question."})
    content = ''
    content += f"Consider the following question:\n {question} \n"
    content += f"Previous answer: {prev_answer} \n"
    #content += f"Review your previous answer and find problems with your answer."
    content += f"{critiques[0]}"
    messages.append({"role":"user", "content":content})
    completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            max_tokens=max_tokens - num_tokens_from_messages(messages),
            messages=messages)
    response = completion["choices"][0]["message"]["content"]
    #print(response)
    messages.append({"role":"assistant", "content":response})
    #messages.append({"role":"user", "content": "Based on the problems you found, improve your answer."})
    messages.append({"role":"user", "content": f"{critiques[1]}"})
    completion2 = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            max_tokens=max_tokens - num_tokens_from_messages(messages),
            messages=messages)
    #print(messages)
    #print(completion["choices"][0]["message"]["content"])
    return completion2["choices"][0]["message"]["content"]
