import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import google.generativeai as genai
import os
import PyPDF2 as pdf
import ollama
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,text, jd):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input+f"Resume: {text}\n Job Description: {jd}")
    return response.text


def get_ollama_response(input, text, jd):
    messages = [
        {"role": "system", "content": input},
        {"role": "user", "content": f"Resume: {text}\nJob Description: {jd}"}
    ]
    model = ollama.chat(model='llama2-uncensored', messages=messages)

    assistant_reply = model.message.content
    return assistant_reply


def input_pdf_text(upload_file):
    reader = pdf.PdfReader(upload_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text+=str(page.extract_text())
    return text

input_prompt = """
You are a highly skilled Application Tracking System (ATS) expert specializing in evaluating resumes for technology-related roles such as Software Engineer, Data Scientist, Data Analyst, and Big Data Engineer.

Task:
- Evaluate the provided resume based on the given job description.
- Provide a percentage match indicating how well the resume aligns with the job description.
- List any **missing keywords** critical for the job.
- Provide a brief **profile summary** based on the resume content.

Output Format:
{
    "JD Match": "XX%",
    "MissingKeywords": [list of missing keywords],
    "Profile Summary": "summary of the resume"
}

Input:
- Resume: {text}
- Job Description: {jd}

Ensure high accuracy and consider the competitive job market when assigning scores and identifying missing keywords.
"""


with st.sidebar:
    st.title("smart Resume evaluator for Resumes")
    st.subheader("About")

    st.title("smart Application Tracking System")
    st.text("Improve your resume ATS")
    jd = st.text_area("Paste the job Description")
    uploaded_file = st.file_uploader("upload your resume",type='pdf',help='please upload the pdf')
    submit = st.button("submit")

if submit:
    if uploaded_file is not None:
        text=input_pdf_text(uploaded_file)
        response = get_ollama_response(input_prompt,text,jd)
        # response = get_gemini_response(input_prompt,text,jd)
        st.subheader(response)
        print(text)