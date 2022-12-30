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
import contractions

def load_model(embedding_size):
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    max_vocab = len(tokenizer.word_index) + 1

    model = tf.keras.models.load_model('chatbot_model')

    encoder_input = model.get_layer('encoder_inputs')
    maxlen_questions = encoder_input.input.shape[1]
    encoder_lstm = model.get_layer('lstm')
    encoder_output = encoder_lstm.output[1:] # only hidden states
    encoder_model = tf.keras.models.Model(encoder_input.input, encoder_output)

    decoder_input = model.get_layer('decoder_inputs')
    maxlen_answers = decoder_input.input.shape[1]
    decoder_hidden_state = tf.keras.layers.Input(shape=(embedding_size,), name='decoder_hidden_state')
    decoder_carry_state = tf.keras.layers.Input(shape=(embedding_size,), name='decoder_carry_state')
    decoder_state_input = [ decoder_hidden_state, decoder_carry_state]
    decoder_embedding = model.get_layer('embedding_1')
    decoder_lstm = model.get_layer('lstm_1')

    decoder_outputs, hidden_state, carry_state = decoder_lstm(decoder_embedding.output, initial_state=decoder_state_input)
    decoder_states = [hidden_state, carry_state]
    decoder_dense = model.get_layer('dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_input.input] + decoder_state_input,  [decoder_outputs] + decoder_states)

    # infer
    idx2word = {value:key for key,value in tokenizer.word_index.items()}

    return encoder_model, decoder_model, tokenizer, idx2word, max_vocab, maxlen_questions, maxlen_answers

def infer(sentence, encoder_model, decoder_model, tokenizer, idx2word, maxlen_questions, maxlen_answers):
    sentence = [sentence]
    preprocessed_sentence = map(lambda x: contractions.fix(x), sentence)
    preprocessed_sentence = tokenizer.texts_to_sequences(sentence)
    preprocessed_sentence = tf.keras.preprocessing.sequence.pad_sequences(preprocessed_sentence , maxlen=maxlen_questions , padding='post')

    state = encoder_model.predict(preprocessed_sentence, verbose=0)
    start_vector = np.zeros((1,1))
    start_vector[0,0] = tokenizer.word_index['bos']
    hidden_state = None
    carry_state = None
    word = ''
    response = ''
    i=0
    confidence = []
    while word!='eos':
        response += word + ' '
        decoder_output, hidden_state, carry_state = decoder_model.predict([start_vector] + state, verbose=0)
        word = idx2word[np.argmax(decoder_output)]
        confidence.append(np.max(decoder_output))
        start_vector[0,0] = tokenizer.word_index[word]
        state = [hidden_state, carry_state]
        i+=1
        if i>=maxlen_answers:
            break
    print(response)
    print(confidence)
    print(np.mean(confidence))
    return response

#encoder_model, decoder_model, tokenizer, idx2word, max_vocab, maxlen_questions, maxlen_answers = load_model(embedding_size)
#infer('hello', encoder_model, decoder_model, tokenizer, idx2word, maxlen_questions, maxlen_answers)