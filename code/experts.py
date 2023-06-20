# Author: Iddo Drori
# Email: idrori@mit.edu

import openai
import pandas as pd
import time
import backoff
import tiktoken
import csv 
import math
from grade import num_tokens_from_messages
import os
from typing import List
from pydantic import BaseModel
import json

openai.api_key = os.getenv('OpenAI_API_Key')

class Experts(BaseModel):
    expert_1: str
    expert_2: str
    expert_3: str

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_experts(department, course_name, question, max_tokens = 8192):
    generic_expert = f"an MIT Professor of {department} teaching the {course_name} course"
    try:              
        response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                temperature=0,
                messages=[
                {"role":"user", "content": "You are " + generic_expert + f". Give an educated guess of who are three experts most capable of solving the following question.\n Question: {question}.\n Return a comma-separated list of three names."}],
                functions=[
                {
                "name": "get_named_experts_for_question",
                "description": "Get named experts for answering a question",
                "parameters": Experts.schema()
                }
                ],
                function_call={"name": "get_named_experts_for_question"}
                )
        arguments_str = response["choices"][0]["message"]["function_call"]["arguments"]
        arguments = json.loads(arguments_str)
        experts = (generic_expert + ', ' + arguments["expert_1"] + ', ' + arguments["expert_2"] + ', ' + arguments["expert_3"])
        return experts
    except openai.error.APIError as e:
        time.sleep(45)
        return get_experts(department, course_name, question)
    except openai.error.APIConnectionError as e:
        time.sleep(45)
        return get_experts(department, course_name, question)
    except openai.error.RateLimitError as e:
        time.sleep(45)
        return get_experts(department, course_name, question)
    except openai.error.Timeout as e:
        time.sleep(45)
        return get_experts(department, course_name, question)
