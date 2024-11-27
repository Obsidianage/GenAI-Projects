import torch
import ollama
import os
import argparse
import json


def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# helper function to get similar context from our data
def get_relevant_context(user_input, embeddings,
                         content, top_k=3):
    """Finds the most relevant lines from the vault based on similarity to the user input."""
    if embeddings.nelement() ==0: # if embeddings are empty
        return []
    # Generate embedding for the user query
    query_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=user_input)['embedding']
    # cosine similarity
    similarity_scores = torch.cosine_similarity(torch.tensor(query_embedding).unsqueeze(0), embeddings)
    # get top-k most relevant results
    top_indices = torch.topk(similarity_scores, k=min(top_k, len(similarity_scores))).indices.tolist()
    # Fetch relevant lines from the vault
    return [content[idx].strip() for idx in top_indices]

def  chat_with_ollama(user_input, context, system_message, conversation_history, model_name):
    """Generates a response using ollama's chat model."""
    # Add user input with context to conversation history
    full_input = user_input + (f"\n\nRelevant Context:\n{context}" if context else "")
    conversation_history.append({"role":"user","content": full_input})

    # call the chat API
    response = ollama.chat(
        model = 'llama2-uncensored',
        messages = [{"role":"system", "content":system_message},
                    *conversation_history]
    )

    # Extract and return the assistant response
    assistant_reply = response.message.content
    conversation_history.append({"role":"assistant","content":assistant_reply})
    return assistant_reply

def main():
    print("Initializing...")

    # Load vault content
    vault_path = "vault.txt"
    if not os.path.exists(vault_path):
        print(f"vault file '{vault_path}' not found. Exiting...")
        return
    
    with open(vault_path, 'r', encoding='utf-8') as file:
        vault_content = file.readlines()

    print("Generatig Embeddings for the vault...")
    vault_embeddings  = []
    for line in vault_content:
        embedding = ollama.embeddings(model='mxbai-embed-large', prompt=line)["embedding"]
        vault_embeddings.append(embedding)
    vault_embeddings_tensor = torch.tensor(vault_embeddings)

    # conversation loop 
    print("starting conversation loop...")

    system_message = "You are a helpful assistant, skilled at extracting information and providing relefant responces."
    conversational_history = []
    model_name = "llama2-uncensored"

    while True:
        user_input = input("Ask a query (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("goodbye!")
            break
        context = "\n".join(get_relevant_context(user_input, vault_embeddings_tensor, vault_content))

        if context:
            print("Relevant Context from vault:\n"+context)
        else:
            print("\nNo relevant context found in the vault.")

        # get response from ollama
        response = chat_with_ollama(user_input, context, system_message, conversational_history, model_name)

        print("\nResponse:\n"+ response)

if __name__=="__main__":
    main()