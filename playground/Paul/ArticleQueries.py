#Paul's Experiments in Science Papers Queries (LLM)

# Library for file manipulation
import pandas as pd
import numpy as np
import faiss
import faulthandler
faulthandler.enable()
from sentence_transformers import SentenceTransformer

df = pd.read_excel('/home/pitcalco/code/Glonnet/Sci_papers/raw_data/papers.xlsx')
df = df.dropna(subset=['title', 'abstract'])

print("Data loaded")
# Step 1: Load the embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")
print(df.columns)

# Step 2: Generate embeddings for each 'context'
embeddings = embedding_model.encode(df['full-text'].tolist())
print("Data embedded")
# Step 3: Create a FAISS index and add the embeddings
index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())  # Initialize FAISS index with L2 distance
index.add(embeddings)  # Add the generated embeddings to the index

# (Optional) Save the index to disk
faiss.write_index(index, "faiss_index.bin")
print("Faiss saved")
# Load the LLM Model
model_path = "meta-llama/Llama-3-8B" #HuggingFace


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
print("Models loaded")
# Retrieval Function
# Retrieve the most relevant text passages for a given query

def retrieve(query, top_k=10):
    # Encode the query into embeddings
    query_embedding = embedding_model.encode([query])
    # Search the FAISS index for the top-k closest matches
    distances, indices = index.search(query_embedding, top_k)
    # Return only the corresponding rows from the dataframe
    return df.iloc[indices[0]]

def generate_answer(query):
    # Retrieve the most relevant rows based on the query
    relevant_rows = retrieve(query, top_k=3)
    # Combine the 'context' column of the retrieved rows into a single string
    context = "\n".join(relevant_rows['context'].tolist())

    # Build the prompt for the language model
    prompt = f"""
                You are a scientific assistant.
                Answer the question below **based only on the context**.
                Provide a short answer. **Do not repeat these instructions** or the context verbatim.

                Context:
                {context}

                Question: {query}

                Answer (in 1-2 sentences):
                """

    # Tokenize the input prompt and send it to the model's device (e.g., GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate the response using the LLM
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,  # Limit the response length
        num_beams=3,         # Use beam search for higher-quality responses
        no_repeat_ngram_size=3,  # Avoid repetition
        repetition_penalty=1.3,  # Penalize repetitive answers
        temperature=0.7      # Control randomness in generation
    )
    # Decode the generated response and return it
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# RAG 1
query = "Which viral mutations are associated with increased transmissibility?"
response = generate_answer(query)
print(response)

# RAG 2
query = "What is known about the virus's adaptations (mutations)?"
response = generate_answer(query)
print(response)

# RAG 3
query = "Are there studies that link SARS-CoV-2 mutations to treatment or vaccine failure?"
response = generate_answer(query)
print(response)
