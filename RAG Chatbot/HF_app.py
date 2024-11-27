import gradio as gr
from datasets import load_dataset

import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig, LlamaTokenizer
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
load_dotenv()


 ## HF_TOKEN is obtained by craeting a .env file in the same directory and placing the Hugging face Token 
 ## format is: HF_TOKEN = "your_token_here"
token = os.environ['HF_TOKEN']
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision='embedded')

data = dataset['train']
data = data.add_faiss_index("embeddings")


model_id = "meta-llama/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id,token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config = bnb_config,
    token = token
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = """You are an assistant for answering questions 
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
"""


def search(query: str, k: int=3):
    embedded_query = ST.encode(query)
    scores, retrieved_examples = data.get_nearest_examples(
        "embeddings", embedded_query,
        k = k
    )

    return scores, retrieved_examples

def format_prompt(prompt, retrieved_documents,k):
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k):
        PROMPT+= f"{retrieved_documents['text'][idx]}\n"

    return PROMPT

@spaces.GPU(duration=150)
def talk(prompt, history):
    k = 3
    scores, retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt, retrieved_documents, k)
    formatted_prompt = formatted_prompt[:2000]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2000
    ).input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    try:
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.75,
            top_p=0.95,
            eos_token_id=terminators
        )

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            print("Output:", "".join(outputs))
            yield "".join(outputs)
    except Exception as e:
        print(f"Error during generation: {e}")



TITLE = "# RAG"

DESCRIPTION = """
A rag pipeline with a chatbot feature
"""

chatbot = gr.ChatInterface(
    fn=talk,
    chatbot=gr.Chatbot(show_label=True, show_copy_button=True),
    theme="Soft",
    examples=[["What's pineapple?"]],
    title=TITLE,
    description=DESCRIPTION,
)
chatbot.launch(debug=True)