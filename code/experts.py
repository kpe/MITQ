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
def get_experts(department, course_name, question, num_experts=3, max_tokens = 8192):
    messages = [{"role":"system", "content":f"You are an MIT Professor of {department} teaching the {course_name} course.\n"},
                {"role":"user", "content": f"Give an educated guess of who are {num_experts} experts most capable of solving the following question.\n Question: {question}.\n Return a comma-separated list of {num_experts} names."}
    ]
    try:
        completion = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0,
                max_tokens=max_tokens - num_tokens_from_messages(messages),
                messages=messages)
        return completion["choices"][0]["message"]["content"]
    except openai.error.APIError as e:
        time.sleep(30)
        return get_experts(department, course_name, question)
    except openai.error.APIConnectionError as e:
        time.sleep(30)
        return get_experts(department, course_name, question)
