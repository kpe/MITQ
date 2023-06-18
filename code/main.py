import openai
import pandas as pd
import time
import backoff
import tiktoken
import csv 
import math
from grade import num_tokens_from_messages, grade
from experts import get_experts
from zero_shot import zero_shot_response
from few_shot import few_shot_response
from self_critique import self_critique_response
import os

openai.api_key = os.getenv('OpenAI_API_Key')


def correct(grade):
    try:
        if ':' in grade:
            return float(grade[grade.index(':')+1:])==5
        else:
            return float(grade) == 5
    except:
        return False


def run_all(input_path, output_path, num_experts = 3, num_fs = 3, most_recent_q = 0):
    df = pd.read_csv(input_path)
    df = df.iloc[most_recent_q:]
    print(len(df.index))
    #question_outputs = {i:[] for i in range(len(df.index))}
    #completed_qs = set()
    with open(output_path, 'a', newline='') as file:
        writer = csv.writer(file)
        first_row = 'Question Index,Department,Course Number,Course Name,Prerequisites,Corequisites,Assignment,Topic,Question Number,Part Number,Percentage of Total Grade,Question Type,Question,Solution Type,Solution,Few shot question 1,Few shot solution 1,Few shot question 2,Few shot solution 2,Few shot question 3,Few shot solution 3'.split(',')
        for i in range(1,num_experts+1):
            first_row.append(f'Expert {i}')
            for j in ['Zero shot', 'Few shot', 'Few shot chain of thought', 'Self-critique 1', 'Self-critique 2']:
                first_row.append(f"{j} response")
                first_row.append(f"{j} grade")
        if most_recent_q==0:
            print(first_row)
            writer.writerow(first_row)

        for index, row in df.iterrows():

            print('Completing question', index)
            question_output = row.values.tolist()
            department = row['Department']
            course_name = row['Course Name']
            question = row['Question']
            solution = row['Solution']
            fs_qs = [[row['Few shot question 1'], row['Few shot solution 1']], [row['Few shot question 2'], row['Few shot solution 2']], [row['Few shot question 3'], row['Few shot solution 3']]]
            experts = get_experts(department, course_name, question, num_experts).split(', ')
            prompts = [lambda expert: zero_shot_response(question, expert), 
                       lambda expert: few_shot_response(expert, question, fs_qs), 
                       lambda expert: few_shot_response(expert, question, fs_qs, True)
            ]
            critiques = [["Review your previous answer and find problems with your answer.", "Based on the problems you found, improve your answer."], ["Please provide feedback on the following incorrect answer.","Given this feedback, answer again."]]
            for expert in experts:
                print("Using expert", expert)
                question_output.append(expert)
                crit = True
                for prompt in [prompts[2]]:
                    prompt_response = prompt(expert) # calls fresh ChatCompletion.create                  
                    prompt_grade = grade(department, course_name, question, solution, prompt_response) # GPT-4 auto-grading comparing answer to solution
                    question_output+=[prompt_response, prompt_grade]
                for critique in critiques:
                    crit_response = self_critique_response(expert, course_name, question, question_output[-2], critique) # calls fresh ChatCompletion.create                  
                    crit_grade = grade(department, course_name, question, solution, crit_response) # GPT-4 auto-grading comparing answer to solution
                    question_output+=[crit_response,crit_grade]

            writer.writerow(question_output)

run_all('MIT_test_set.csv', 'MIT_test_set_graded.csv')
