import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks,embedding=OllamaEmbeddings(model="llama2-uncensored"))
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """answer the questions\n\n
    context:\n {context}?\n
    question: \n{question}\n

    answer:    
    """

    model = Ollama(model='llama2-uncensored')

    prompt = PromptTemplate(template = prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):

    new_db = FAISS.load_local("faiss_index",
                              OllamaEmbeddings(model="llama2-uncensored"),
                              allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response = chain(
        {"input_documents":docs,"question":user_question},
        return_only_outputs=True)
    
    print(response)
    st.write(response["output_text"])


def main():
    st.set_page_config("Multi PDF chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's - Chat Agent")

    user_question = st.text_input("ask a question from the pdf files uploaded")

    if user_question:
        user_input(user_question)

    with st.sidebar:

        st.title("PDF files section")
        pdf_docs = st.file_uploader("upload your pdf files & \n click on the submit & process button", accept_multiple_files=True)
        if st.button("submit & process"):
            with st.spinner("preprocessing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.write("---")
    

if __name__=="__main__":
    main()