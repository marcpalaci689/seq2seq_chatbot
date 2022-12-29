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
import json

root_dir = Path(__file__).parent.absolute()

# first prepare the WikiQA data
data_dir = os.path.join(root_dir, 'WikiQACorpus')
print(data_dir)
questions = []
answers = []
filename= 'WikiQA-train.txt'
questions=[]
answers=[]
with open(os.path.join(data_dir,filename), 'r', encoding='utf-8') as file:
    for line in file:
        data = line.split('\t')
        question, answer = data[0], data[1]
        if len(questions)>=1 and question == questions[-1]:
            continue
        else:
            questions.append(question)
            answers.append(answer)

# prepare the chatbot data
data_dir = os.path.join(root_dir, 'chatbot_data')
print(data_dir)
for filename in os.listdir(data_dir):
    print(filename)
    with open(os.path.join(data_dir,filename), 'r', encoding='utf-8') as file:
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

# prepare the convai3d data
data_dir = os.path.join(root_dir, 'convai3_data')
with open(os.path.join(data_dir,'data_tolokers.json'), 'r', encoding='utf-8') as file:
    data = json.load(file)
for conv in data:
    dialogue = conv['dialog']
    if len(dialogue) >= 2:
        while len(dialogue)>1:
            # keep popping front until a convo is initiated
            if dialogue[0]['sender_class'] == dialogue[1]['sender_class']:
                dialogue.pop(0)
                continue
            else:
                q = re.sub("[\(\[].*?[\)\]]", "", dialogue[0]['text'])
                a = re.sub("[\(\[].*?[\)\]]", "", dialogue[1]['text'])
                q = re.sub("[^A-Za-z0-9 -]", "", q).rstrip()
                a = re.sub("[^A-Za-z0-9 -]", "", a).rstrip()
                if len(a.split()) == 1 or len(q.split()) == 1:
                    pass
                else:
                    questions.append(q)
                    answers.append(a)
                dialogue.pop(0)
                dialogue.pop(0)

answers_with_tags = []
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)

answers = ['<BOS> ' + answer + ' <EOS>' for answer in answers_with_tags]

# save prepared data into text files
data_dir = os.path.join(root_dir, 'prepared_data')
with open(os.path.join(data_dir, 'questions.txt'), 'w', encoding='utf-8') as file:
    for question in questions:
        file.write(question)
        file.write('\n')

with open(os.path.join(data_dir, 'answers.txt'), 'w', encoding='utf-8') as file:
    for answer in answers:
        file.write(answer)
        file.write('\n')



    
