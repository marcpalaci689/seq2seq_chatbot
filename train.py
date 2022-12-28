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
data_dir = os.path.join(root_dir, 'prepared_data')
questions= []
answers = []
with open(os.path.join(data_dir, 'questions.txt'), 'r') as file:
    for line in file:
        questions.append(line.rstrip('\n'))

with open(os.path.join(data_dir, 'answers.txt'), 'r') as file:
    for line in file:
        answers.append(line.rstrip('\n'))


#train the tokenizer on the input and output data and save
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
with open('tokenizer100.pkl', 'wb') as file:
    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
max_vocab = len(tokenizer.word_index) + 1

# create a GLOVE embedding matrix based off the tokenizer vocabulary
glove_dir = os.path.join(os.path.split(root_dir)[0], 'glove.6B/glove.6B.100d.txt')
embedding = np.zeros((max_vocab, 100))
with open(glove_dir, 'r') as file:
    for line in file:
        data = line.split()
        word, vec = data[0], data[1:]
        if word in tokenizer.word_index:
            embedding[tokenizer.word_index[word],:] = vec

# convert questions to tokenized/padded form
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = np.max([len(x) for x in tokenized_questions])
padded_questions = tf.keras.preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
padded_questions = np.array(padded_questions)

# conver answers to tokenized/padded form
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = np.max([len(x) for x in tokenized_answers])
padded_answers = tf.keras.preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
padded_answers = np.array(padded_answers)


# create the desired output data as 1 hot encoded
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_output = tf.keras.preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
#target_output = tf.keras.utils.to_categorical(padded_output, max_vocab)
#target_output=np.array(target_output)
target_output=np.array(padded_output)


encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions,), name='encoder_inputs')
encoder_embedding = tf.keras.layers.Embedding(max_vocab, 100, mask_zero=True, embeddings_initializer=tf.keras.initializers.Constant(embedding), trainable=True) (encoder_inputs) #TODO add initializer to embedding
_, encoder_hidden_state, encoder_carry_state = tf.keras.layers.LSTM(100, return_state=True) (encoder_embedding)
encoder_state = [encoder_hidden_state, encoder_carry_state]

decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers,), name='decoder_inputs')
decoder_embedding = tf.keras.layers.Embedding(max_vocab, 100, mask_zero=True, embeddings_initializer=tf.keras.initializers.Constant(embedding), trainable=True) (decoder_inputs) #TODO add initializer to embedding
decoder_outputs, _, _ = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True) (decoder_embedding, initial_state=encoder_state)
outputs = tf.keras.layers.Dense(max_vocab, activation=tf.keras.activations.softmax) (decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit([padded_questions, padded_answers], target_output, batch_size=128, epochs=300)
model.save('chatbot_model100') 