import requests
import streamlit as st


def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/letter/invoke",
        json={'input':{'topic':input_text}})
    return response.json()['output']

## streamlit framework

st.title("Diddys party")
input_text=st.text_input("We sell Diddy shampoo")

st.write(get_ollama_response(input_text))

