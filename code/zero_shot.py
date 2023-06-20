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
def zero_shot_response(system, question, cot=True, max_tokens=8192):
    try:
        messages = [
            {"role": "system", "content": f"You are {system}\n"
                                          f"Your task is to answer the following question."}                       
          ]
        content = f"Please answer the following question.\nQuestion: {question}\n"
        if cot:
            content += "Let's think step by step."
        messages.append({"role":"user", "content":content})
        completion = openai.ChatCompletion.create(
            model=os.getenv('Prompt_Engine'),
            temperature=0,
            max_tokens=max_tokens - num_tokens_from_messages(messages),
            messages=messages)
        return completion["choices"][0]["message"]["content"]
    except openai.error.APIError as e:
        time.sleep(45)
        return zero_shot_response(system, question)
    except openai.error.APIConnectionError as e:
        time.sleep(45)
        return zero_shot_response(system, question)
    except openai.error.RateLimitError as e:
        time.sleep(45)
        return zero_shot_response(system, question)
    except openai.error.Timeout as e:
        time.sleep(45)
        return zero_shot_response(system, question)
