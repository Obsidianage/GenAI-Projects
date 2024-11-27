import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Splits the text into manageable chunks for vectorization."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Generates a FAISS vector store from text chunks."""
    vector_store = FAISS.from_texts(
        text_chunks, embedding=OllamaEmbeddings(model="llama2-uncensored")
    )
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Sets up the conversational chain with a custom prompt."""
    prompt_template = """
    Use the following context to answer the question:
    Context: {context}
    Question: {question}
    Answer:
    """
    model = Ollama(model="llama2-uncensored")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def process_pdfs(pdf_files):
    """Processes uploaded PDFs to build a FAISS index."""
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return "PDFs processed and indexed successfully!"


def ask_question(user_question):
    """Handles user questions by querying the FAISS vector store."""
    vector_store = FAISS.load_local(
        "faiss_index",
        OllamaEmbeddings(model="llama2-uncensored"),
        allow_dangerous_deserialization=True,  # Enable deserialization for local files
    )
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


def main():
    """Main Gradio Interface."""
    with gr.Blocks() as demo:
        gr.Markdown("### Multi-PDF Chat Agent using Gradio")
        
        with gr.Row():
            pdf_input = gr.File(
                label="Upload your PDF files",
                file_types=[".pdf"],
                file_count="multiple",
            )
            process_button = gr.Button("Process PDFs")

        output_status = gr.Textbox(label="Processing Status", interactive=False)

        process_button.click(
            fn=process_pdfs,
            inputs=[pdf_input],
            outputs=[output_status],
        )

        user_question = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        answer_box = gr.Textbox(label="Chatbot Reply", interactive=False)

        user_question.submit(fn=ask_question, inputs=[user_question], outputs=[answer_box])

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
