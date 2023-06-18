import openai
import pandas as pd
import time
import backoff
import tiktoken
import csv 
import math
import os

openai.api_key = os.getenv('OpenAI_API_Key')

def num_tokens_from_messages(messages, model=None):
    if model is None:
        model = os.getenv('Grading_Engine')
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def grade(department, course_name, question, solution, answer, max_tokens=8192): # 0-5
    try:
        messages = [
            {"role": "system", "content": f"You are an MIT Professor of {department} teaching the {course_name} course."
                                          f"Your task is to grade the answer to a question based on a provided solution."},
            {"role": "user", "content": f"Question: {question}\n" +
                                        f"Solution: {solution}\n" +
                                        f"Please grade the following answer.\n" + 
                                        f"Answer: {answer}\n" +
                                        f"Assign a score for the answer on a scale of 0 to 5, where 0 is the lowest and 5 is the highest based on correctness. Do not provide any additional feedback or explanation."}
          ]
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            max_tokens=max_tokens - num_tokens_from_messages(messages),
            messages=messages)
        return completion["choices"][0]["message"]["content"]
    except openai.error.APIError as e:
        print('error received')
        time.sleep(45)
        return grade(department, course_name, question, solution, answer)
    except openai.error.APIConnectionError as e:
        print('connection error received')
        time.sleep(45)
        return grade(department, course_name, question, solution, answer)

