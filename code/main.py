# Authors: Sam Florin, Iddo Drori
# Emails: sflorin@mit.edu, idrori@mit.edu

import openai
import pandas as pd
import time
import backoff
import tiktoken
import csv 
import math
import logging
from grade import num_tokens_from_messages, grade
from experts import get_experts
from zero_shot import zero_shot_response
from few_shot import few_shot_response
from self_critique import self_critique_response
import os

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['OpenAI_API_Key'] = 'sk-' # insert here
os.environ['Prompt_Engine'] = 'gpt-4-0613' # or if you don't have GPT-4 access replace 'gpt-4-0613' with 'gpt-3.5-turbo-16k'
os.environ['Grading_Engine'] = 'gpt-4-0613' # auto-grading compares with ground truth solution 
os.environ['Experts_Engine'] = 'gpt-4-0613' # for function calling
openai.api_key = os.getenv('OpenAI_API_Key')

def correct(grade):
    try:
        if ':' in grade:
            return float(grade[grade.index(':')+1:])==5
        else:
            return float(grade) == 5
    except:
        return False


def run_all(input_path, output_path, num_fs = 3, most_recent_q = 0): 
    df = pd.read_csv(input_path)
    df = df.iloc[most_recent_q:]
    print(len(df.index))
    #question_outputs = {i:[] for i in range(len(df.index))}
    #completed_qs = set()
    with open(output_path, 'a', newline='') as file:
        writer = csv.writer(file)
        first_row = 'Question Index,Department,Course Number,Course Name,Prerequisites,Corequisites,Assignment,Topic,Question Number,Part Number,Percentage of Total Grade,Question Type,Question,Solution Type,Solution,Few shot question 1,Few shot solution 1,Few shot question 2,Few shot solution 2,Few shot question 3,Few shot solution 3'.split(',')
        for i in range(1,5):
            first_row.append(f'Expert {i}')
            for j in ['Zero shot', 'Few shot']: # default CoT prompt, 'Self-critique'
                first_row.append(f"{j} response")
                first_row.append(f"{j} grade")
        if most_recent_q==0:
            print(first_row)
            writer.writerow(first_row)

        logging.info("Starting to grade questions")
        for index, row in df.iterrows():
            logging.info(f"Starting to grade question {index}")
            print('Completing question', index)
            question_output = row.values.tolist()
            department = row['Department']
            course_name = row['Course Name']
            question = row['Question']
            solution = row['Solution']
            fs_qs = [[row['Few shot question 1'], row['Few shot solution 1']], [row['Few shot question 2'], row['Few shot solution 2']], [row['Few shot question 3'], row['Few shot solution 3']]]
            experts = get_experts(department, course_name, question).split(', ')
            prompts = [lambda expert: zero_shot_response(expert, question, True), 
                       #lambda expert: few_shot_response(expert, question, fs_qs), 
                       lambda expert: few_shot_response(expert, question, fs_qs, True)
            ]
            critique = ["Review your previous answer and find problems with your answer.", "Based on the problems you found, improve your answer."] # never use that a question was wrong
            for expert in experts: # generic, named 1, 2, 3
                logging.info(f"Starting to grade question {index} with expert {expert}")
                print("Using expert", expert)
                question_output.append(expert)
                crit = False
                for prompt in prompts:
                    logging.info(f"Starting to grade question {index} with expert {expert} using prompt\n {prompt}")
                    prompt_response = prompt(expert) # calls fresh ChatCompletion.create, never use solution 
                    logging.info(f"Prompt response: {prompt_response}")
                    prompt_grade = grade(department, course_name, question, solution, prompt_response) # GPT-4 auto-grading comparing answer to solution
                    logging.info(f"Prompt grade: {prompt_grade}")
                    question_output+=[prompt_response, prompt_grade]
                if (crit):
                    crit_response = self_critique_response(expert, course_name, question, question_output[-2], critique) # calls fresh ChatCompletion.create, never use solution                  
                    crit_grade = grade(department, course_name, question, solution, crit_response) # GPT-4 auto-grading comparing answer to solution
                    question_output+=[crit_response,crit_grade]

            writer.writerow(question_output) # + human meta-grading of GPT-4 auto-grading

file_name = 'mit_test.csv'
current_directory = os.path.dirname(os.path.abspath(__file__))
full_file_path = os.path.join(current_directory, file_name)
run_all(full_file_path, 'mit_test_graded.csv', most_recent_q = 0) # apply same methods for all questions zs+fs+cot+critique+expert
