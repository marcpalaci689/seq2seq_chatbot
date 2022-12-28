import os
import yaml
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras import preprocessing
from gensim.models import Word2Vec
import tensorflow as tf
import re
import numpy as np
import pickle

root_dir = Path(__file__).parent.absolute()

# first prepare the WikiQA data
data_dir = os.path.join(root_dir, 'WikiQACorpus')
print(data_dir)
questions = []
answers = []
filename= 'WikiQA-train.txt'
questions=[]
answers=[]
with open(os.path.join(data_dir,filename), 'r') as file:
    for line in file:
        data = line.split('\t')
        question, answer = data[0], data[1]
        if len(questions)>=1 and question == questions[-1]:
            continue
        else:
            questions.append(question)
            answers.append(answer)

data_dir = os.path.join(root_dir, 'chatbot_data')
print(data_dir)
for filename in os.listdir(data_dir):
    print(filename)
    with open(os.path.join(data_dir,filename), 'r') as file:
        data = yaml.safe_load(file)
        for conversation in data['conversations']:
            question = conversation[0]
            answer = conversation[1]
            if len(questions)>=1 and question == questions[-1]:
                continue
            else:
                #if len(conversation)>2: #should we be doing this?
                #    for response in conversation[2:]:
                #        answer += ' {0}'.format(response)
                questions.append(question)
                answers.append(answer)

answers_with_tags = []
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)

answers = ['<BOS> ' + answer + ' <EOS>' for answer in answers_with_tags]

# save prepared data into text files
data_dir = os.path.join(root_dir, 'prepared_data')
with open(os.path.join(data_dir, 'questions.txt'), 'w') as file:
    for question in questions:
        file.write(question)
        file.write('\n')

with open(os.path.join(data_dir, 'answers.txt'), 'w') as file:
    for answer in answers:
        file.write(answer)
        file.write('\n')



    
