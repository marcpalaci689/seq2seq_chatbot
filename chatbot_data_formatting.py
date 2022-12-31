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
questions=[]
answers=[]

# first prepare the WikiQA data
""" data_dir = os.path.join(root_dir, 'WikiQACorpus')
print(data_dir)
filename= 'WikiQA-train.txt'
with open(os.path.join(data_dir,filename), 'r', encoding='utf-8') as file:
    for line in file:
        data = line.split('\t')
        question, answer = data[0], data[1]
        if len(questions)>=1 and question == questions[-1]:
            continue
        else:
            questions.append(question)
            answers.append(answer) """

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

# Prepare the convo data                
data_dir = os.path.join(root_dir, 'convo_data/convo1')
for filename in os.listdir(data_dir):
    path = os.path.join(data_dir, filename)
    print(path)
    with open(path, 'r', encoding='utf-8') as file:
        i=0
        for line in file:
            split = line.split(':')
            if len(split)>1:
                sentence = split[1].rstrip('\n')
                sentence = sentence.lstrip(' ')
                if sentence.startswith('"') and sentence.endswith('"'):
                    sentence = sentence[1:-1]
                if i%2 == 0:
                    #print('q) {0}'.format(split[1]))
                    questions.append(sentence)
                else:
                    #print('a) {0}'.format(split[1]))
                    answers.append(sentence)
                i+=1
        if i%2==1:
            questions.pop(-1)
            
data_dir = os.path.join(root_dir, 'convo_data/convo2')
for filename in os.listdir(data_dir):
    path = os.path.join(data_dir, filename)
    print(path)
    with open(path, 'r', encoding='utf-8') as file:
        i=0
        for line in file:
            split = line.split(':')
            if len(split)>1:
                sentence = split[1].rstrip('\n')
                sentence = sentence.lstrip(' ')
                if sentence.startswith('"') and sentence.endswith('"'):
                    sentence = sentence[1:-1]
                if i%2 == 0:
                    #print('q) {0}'.format(split[1]))
                    questions.append(sentence)
                else:
                    #print('a) {0}'.format(split[1]))
                    answers.append(sentence)
                i+=1
        if i%2==1:
            questions.pop(-1)

# prepare squad data set
""" data_dir = os.path.join(root_dir, 'squad_data/train-v2.0.json')
with open(data_dir, 'r', encoding='utf-8') as file:
    data = json.load(file)
for entry in data['data']:
    for paragraph in entry['paragraphs']:
        for qas in paragraph['qas']:
            if qas['question'] and qas['answers']:
                questions.append(qas['question'])
                answers.append(qas['answers'][0]['text']) """

# prepare the convai3d data
""" 
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
 """
answers_with_tags = []
questions_with_tags = []
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
        questions_with_tags.append(questions[i])
    else:
        print(questions[i])
        print(answers[i])
        print(' ')


questions = []
answers = []
# remove questions of 1 token or less
for i in range(len(questions_with_tags)):
    if len(questions_with_tags[i].split())>1:
        questions.append(questions_with_tags[i])
        answers.append('<BOS> ' + answers_with_tags[i] + ' <EOS>')
        

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



data_dir = os.path.join(root_dir, 'squad_data/train-v2.0.json')
with open(data_dir, 'r', encoding='utf-8') as file:
    data = json.load(file)
    i=0
for entry in data['data']:
    for paragraph in entry['paragraphs']:
        for qas in paragraph['qas']:
            if qas['question'] and qas['answers']:
                questions.append(qas['question'])
                answers.append(qas['answers'][0]['text'])
        


    
