import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
import os
import ollama

from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = """
Summarize the given transcript text. Provide the summary in **bullet points**, ensuring it highlights the key points in a professional and concise manner. Limit the summary to 250 words. Here is the transcript text:
"""


## getting the transcript data from yt video
def extract_trascript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]

        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript
    except Exception as e:
        raise e

def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

def generate_ollama_content(transcript_text, prompt):
    # Format the messages with role and content
    messages = [
        {"role": "system", "content": "You are an expert summarizer tasked with summarizing video transcripts concisely. Ensure the summary is clear, professional, and highlights the key points in bullet format."},
        {"role": "user", "content": prompt + transcript_text},
    ]

    # Call Ollama's chat API
    response = ollama.chat(model='llama2-uncensored', messages=messages)

    # Extract the assistant's reply
    assistant_reply = response.message.content  # Ensure you handle the API's response format correctly
    return assistant_reply



st.title("youtube transcript to detailed notes converter")
youtube_link = st.text_input("enter youtube video link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Notes"):
    transcript_text = extract_trascript_details(youtube_link)

    if transcript_text:
        # summary=generate_ollama_content(transcript_text,prompt)
        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("## Detailed notes:")
        st.write(summary)
        print("\n")
        print(transcript_text)