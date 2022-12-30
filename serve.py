from fastapi import FastAPI
from infer import load_model, infer
from search import search
from pathlib import Path
import os

app = FastAPI()
embedding_size=50
root_dir = Path(__file__).parent.absolute()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
async def startup_event():
    global encoder_model, decoder_model, tokenizer, idx2word, max_vocab, maxlen_questions, maxlen_answers 
    encoder_model, decoder_model, tokenizer, idx2word, max_vocab, maxlen_questions, maxlen_answers = load_model(embedding_size)
    return

@app.post("/infer/")
async def infer_model(sentence):
    print(sentence)
    response = infer(sentence, encoder_model, decoder_model, tokenizer, idx2word, maxlen_questions, maxlen_answers)
    return response

@app.post('/feedback/')
async def provide_feedback(question, answer):
    data_dir = os.path.join(root_dir, 'feedback_data/data.txt')
    with open(data_dir, 'a', encoding='utf-8') as file:
        file.write(question + '\t' + answer + '\n')
    return

@app.get('/search/')
async def search_google(question):
    answer = search(question)
    return answer


