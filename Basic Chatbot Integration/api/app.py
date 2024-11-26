from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ollama as ola
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title = "Rizzler Server",
    version = '1.0',
    description="A nuclear project"
)

llm = Ollama(model = "llama2-uncensored")

prompt1 = ChatPromptTemplate.from_template("write a letter to Adolf Hitler for {topic} ")

add_routes(
    app,
    prompt1|llm,
    path="/letter"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

