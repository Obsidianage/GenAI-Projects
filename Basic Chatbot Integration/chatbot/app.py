from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a good assistant. So respond aggresivley"),
        ("user","Question:{question}")
    ]
)

# Streamlit framwork

st.title("langchain demo with some model uk")
input_text = st.text_input("search the hitler version you want")

llm = Ollama(model ="llama2-uncensored")
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))