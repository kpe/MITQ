import openai
import pandas as pd
import time
import backoff
import tiktoken
import csv 
import math
from grade import num_tokens_from_messages

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def zero_shot_response(system, question, max_tokens=8192):
    try:
        messages = [
            {"role": "system", "content": f"You are {system}\n"
                                          f"Your task is to answer the following question."},
            {"role": "user", "content": f"Please answer the following question.\n" + 
                                        f"Question: {question}\n" 
            }                               
          ]

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            max_tokens=max_tokens - num_tokens_from_messages(messages),
            messages=messages)
        return completion["choices"][0]["message"]["content"]
    except openai.error.APIError as e:
        time.sleep(30)
        return zero_shot_response(system, question)
    except openai.error.APIConnectionError as e:
        time.sleep(30)
        return zero_shot_response(system, question)
